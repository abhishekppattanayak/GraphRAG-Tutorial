import os
import shutil
from typing import List, Dict, Optional
from enum import Enum
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from groq import Groq
from neo4j import GraphDatabase
import spacy
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=True)

# Configuration Management
class Settings(BaseSettings):
    GROQ_API_KEY: str
    NEO4J_URI: str
    NEO4J_USERNAME: str
    NEO4J_PASSWORD: str
    GROQ_MODEL: str = "deepseek-r1-distill-llama-70b"
    UPLOAD_DIR: str = "uploads"
    
    ALLOWED_ENTITY_TYPES: List[str] = Field(default=[
        "concept", "chapter", "subject", "formula", 
        "diagram", "theorem", "example", "exercise",
        "activity", "experiment", "definition",
        "historical_figure", "scientist", "mathematician",
        "table", "equation"
    ])
    ALLOWED_RELATIONSHIPS: List[str] = Field(default=[
        "part_of", "related_to", "depends_on", 
        "applies_to", "proved_by", "illustrated_by",
        "authored_by", "appears_in", "prerequisite_for"
    ])
    
    class Config:
        env_file = ".env"

# Pydantic Models
class EntityType(str, Enum):
    CONCEPT = "concept"
    CHAPTER = "chapter"
    SUBJECT = "subject"
    FORMULA = "formula"
    DIAGRAM = "diagram"
    THEOREM = "theorem"
    EXAMPLE = "example"
    EXERCISE = "exercise"
    ACTIVITY = "activity"
    EXPERIMENT = "experiment"
    DEFINITION = "definition"
    HISTORICAL_FIGURE = "historical_figure"
    SCIENTIST = "scientist"
    MATHEMATICIAN = "mathematician"
    TABLE = "table"
    EQUATION = "equation"

class Entity(BaseModel):
    name: str
    type: EntityType
    properties: Dict[str, str] = Field(default_factory=dict)
    
    def validate_entity_type(cls, v):
        if v not in Settings().ALLOWED_ENTITY_TYPES:
            raise ValueError(f"Invalid entity type: {v}. Allowed types: {Settings().ALLOWED_ENTITY_TYPES}")
        return v

class Relationship(BaseModel):
    source: str
    target: str
    type: str
    properties: Dict[str, str] = Field(default_factory=dict)
    
    def validate_relationship_type(cls, v):
        if v not in Settings().ALLOWED_RELATIONSHIPS:
            raise ValueError(f"Invalid relationship type: {v}. Allowed types: {Settings().ALLOWED_RELATIONSHIPS}")
        return v

class KnowledgeGraphData(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]
    source_text: Optional[str] = None
    book_reference: Optional[str] = None
    grade_level: Optional[str] = None

class BookUploadRequest(BaseModel):
    book_reference: str
    grade_level: Optional[str] = None
    subject: Optional[str] = None

class ProcessingStatus(BaseModel):
    job_id: str
    status: str
    message: str
    timestamp: str
    progress: Optional[float] = None
    total_chunks: Optional[int] = None
    processed_chunks: Optional[int] = None

class QueryRequest(BaseModel):
    query: str
    limit: int = 10

# Knowledge Graph Builder
class NCERTKnowledgeGraphBuilder:
    def __init__(self):
        self.settings = Settings()
        self.groq_client = Groq(api_key=self.settings.GROQ_API_KEY)
        self.nlp = spacy.load("en_core_web_sm")
        self.driver = GraphDatabase.driver(
            self.settings.NEO4J_URI,
            auth=(self.settings.NEO4J_USERNAME, self.settings.NEO4J_PASSWORD)
        )
        # Create upload directory if it doesn't exist
        os.makedirs(self.settings.UPLOAD_DIR, exist_ok=True)
    
    def extract_kg_elements(self, text: str, context: Optional[Dict] = None) -> Dict:
        """Extract KG elements with schema enforcement through prompt only"""
        prompt = self._build_prompt(text, context)
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.settings.GROQ_MODEL,
                response_format={"type": "json_object"},
                temperature=0.1  # Lower temperature for more consistent formatting
            )
            
            kg_data = json.loads(response.choices[0].message.content)
            
            # Basic cleanup without strict validation
            if 'relationships' in kg_data:
                for rel in kg_data['relationships']:
                    # Ensure 'type' field exists
                    if 'relation' in rel:
                        rel['type'] = rel.pop('relation')
                    
            if context:
                kg_data.update(context)
                
            return kg_data
            
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON response from Groq API")
        except Exception as e:
            raise ValueError(f"Failed to process KG data: {str(e)}")
    
    def _build_prompt(self, text: str, context: Optional[Dict]) -> str:
        """Construct a strict prompt that enforces the schema through instructions"""
        context_info = ""
        if context:
            context_info = (
                f"\nAdditional Context:\n"
                f"- Book: {context.get('book_reference', 'unknown')}\n"
                f"- Grade: {context.get('grade_level', 'unknown')}\n"
                f"- Subject: {context.get('subject', 'unknown')}\n"
            )
        
        return f"""
        Extract knowledge graph elements from NCERT textbook content following these STRICT rules:

        1. OUTPUT FORMAT (MUST follow exactly):
        {{
            "entities": [
                {{
                    "name": "entity_name",
                    "type": "allowed_type",
                    "properties": {{
                        "key": "value"
                    }}
                }}
            ],
            "relationships": [
                {{
                    "source": "source_entity_name",
                    "target": "target_entity_name",
                    "type": "relationship_type",
                    "properties": {{
                        "key": "value"
                    }}
                }}
            ]
        }}

        2. ENTITY TYPES (must use exactly these):
        - concept, chapter, subject, formula, diagram, theorem, 
        - example, exercise, activity, experiment, definition,
        - historical_figure, scientist, mathematician, table, equation

        3. RELATIONSHIP TYPES (must use exactly these):
        - part_of, related_to, depends_on, applies_to, proved_by,
        - illustrated_by, authored_by, appears_in, prerequisite_for,
        - instance_of, method_of, component_of, example_of

        4. SPECIAL INSTRUCTIONS:
        - Always include both "source" and "target" in relationships
        - Use "type" field for relationship type (not "relation")
        - For questions → type="exercise"
        - For figures/diagrams → type="diagram"
        - For math expressions → type="formula"

        {context_info}

        Textbook content to analyze:
        {text}

        Return ONLY the JSON output matching the exact format above.
        
            Extract knowledge graph elements with these additional rules:
    
        5. LABEL SIMPLIFICATION RULES:
        - Use simple, single-word labels where possible
        - For complex concepts: 
            - "chemical compound" → "compound"
            - "periodic table element" → "element"
            - "scientific phenomenon" → "phenomenon"
        - Replace spaces with underscores
        """
    
    def store_in_neo4j(self, kg_data: Dict):
        """Store extracted knowledge graph in Neo4j"""
        with self.driver.session() as session:
            # Create entities (nodes)
            for entity in kg_data.get("entities", []):
                session.execute_write(
                    self._create_entity_node,
                    entity.get("name", ""),
                    entity.get("type", "concept"),  # Default to 'concept' if missing
                    entity.get("properties", {})
                )
            
            # Create relationships
            for rel in kg_data.get("relationships", []):
                session.execute_write(
                    self._create_relationship,
                    rel.get("source", ""),
                    rel.get("target", ""),
                    rel.get("type", "related_to"),  # Default to 'related_to' if missing
                    rel.get("properties", {})
                )
            
            # Add metadata if available
            if kg_data.get("source_text") or kg_data.get("book_reference"):
                session.execute_write(
                    self._add_metadata,
                    kg_data.get("source_text"),
                    kg_data.get("book_reference"),
                    kg_data.get("grade_level")
                )
    
    def _create_entity_node(self, tx, name: str, label: str, properties: Dict):
        """Create a node with sanitized inputs"""
        # Sanitize inputs
        safe_label = self._sanitize_neo4j_input(label)
        safe_name = self._sanitize_neo4j_input(name)
        
        # Sanitize property keys and values
        safe_properties = {
            self._sanitize_neo4j_input(k): self._sanitize_neo4j_input(str(v))
            for k, v in properties.items()
        }
        
        props_str = ", ".join([f"{k}: ${k}" for k in safe_properties.keys()])
        query = (
            f"MERGE (n:`{safe_label}` {{name: $name"
            f"{', ' + props_str if props_str else ''}}})"
            f" SET n.created_at = datetime()"
        )
        params = {"name": safe_name, **safe_properties}
        tx.run(query, **params)

    def _create_relationship(self, tx, source: str, target: str, rel_type: str, properties: Dict):
        """Create a relationship with sanitized inputs"""
        safe_rel_type = self._sanitize_neo4j_input(rel_type)
        safe_source = self._sanitize_neo4j_input(source)
        safe_target = self._sanitize_neo4j_input(target)
        
        safe_properties = {
            self._sanitize_neo4j_input(k): self._sanitize_neo4j_input(str(v))
            for k, v in properties.items()
        }
        
        props_str = ", ".join([f"{k}: ${k}" for k in safe_properties.keys()])
        query = (
            f"MATCH (a), (b) "
            f"WHERE a.name = $source AND b.name = $target "
            f"MERGE (a)-[r:`{safe_rel_type}` "
            f"{{{'{'} + props_str + {'}'} if props_str else ''}}]->(b)"
            f" SET r.created_at = datetime()"
        )
        params = {"source": safe_source, "target": safe_target, **safe_properties}
        tx.run(query, **params)
        
    def _add_metadata(self, tx, source_text: Optional[str], book_ref: Optional[str], grade: Optional[str]):
        """Add processing metadata to the graph"""
        if book_ref:
            tx.run(
                "MERGE (m:Metadata {book_reference: $book_ref}) "
                "SET m.last_updated = datetime(), "
                "m.grade_level = $grade, "
                "m.processed = true",
                book_ref=book_ref,
                grade=grade
            )
    
    def process_material(self, text: str, context: Optional[Dict] = None) -> Dict:
        """Full processing pipeline that returns raw dictionary"""
        # Preprocessing
        doc = self.nlp(text)
        clean_text = " ".join([
            token.lemma_.lower() for token in doc 
            if not token.is_stop and not token.is_punct
        ])
        
        # Knowledge extraction
        kg_data = self.extract_kg_elements(clean_text, context)
        
        # Ensure basic structure exists
        kg_data.setdefault("entities", [])
        kg_data.setdefault("relationships", [])
        
        # Store in Neo4j
        self.store_in_neo4j(kg_data)
        
        return kg_data
    
    def load_and_process_pdf(self, file_path: str, context: Dict, job_id: str = None, status_callback=None) -> List[Dict]:
        """Load and process a single NCERT PDF file with status updates"""
        try:
            # Load PDF
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            # Split text into manageable chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_documents(pages)
            
            # Process each chunk
            results = []
            total_chunks = len(chunks)
            
            for i, chunk in enumerate(chunks):
                result = self.process_material(chunk.page_content, context)
                results.append(result)
                
                # Update progress if callback is provided
                if status_callback and job_id:
                    progress = (i + 1) / total_chunks * 100
                    status_callback(job_id, "processing", f"Processed chunk {i+1}/{total_chunks}", 
                                   progress, total_chunks, i+1)
            
            # Final status update
            if status_callback and job_id:
                status_callback(job_id, "completed", f"Completed processing {total_chunks} chunks", 
                               100.0, total_chunks, total_chunks)
                
            return results
        
        except Exception as e:
            # Update status on error
            if status_callback and job_id:
                status_callback(job_id, "failed", f"Error: {str(e)}", 0, 0, 0)
            raise ValueError(f"Error processing PDF {file_path}: {str(e)}")

    def _sanitize_neo4j_input(self, input_str: str) -> str:
        """Sanitize strings for Neo4j labels and properties"""
        if not input_str:
            return "unknown"
        
        # Remove special characters and replace spaces
        sanitized = "".join(
            c if c.isalnum() or c in "_-" else "_" 
            for c in input_str.strip()
        )

        # Ensure it starts with a letter
        if sanitized and not sanitized[0].isalpha():
            sanitized = "e_" + sanitized
            
        return sanitized[:64]  # Limit length
    
    def query_knowledge_graph(self, query: str, limit: int = 10) -> Dict:
        """Query the knowledge graph using natural language"""
        try:
            # Convert natural language query to Cypher using Groq
            cypher_prompt = f"""
            Convert the following natural language question into a Neo4j Cypher query.
            The knowledge graph contains nodes with types like: concept, chapter, subject, formula, 
            diagram, theorem, example, exercise, activity, experiment, definition,
            historical_figure, scientist, mathematician, table, equation.
            
            Relationships include: part_of, related_to, depends_on, applies_to, proved_by,
            illustrated_by, authored_by, appears_in, prerequisite_for.
            
            Return ONLY the Cypher query without any explanation or markdown.
            
            Question: {query}
            """
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": cypher_prompt}],
                model=self.settings.GROQ_MODEL,
                temperature=0.1
            )
            
            cypher_query = response.choices[0].message.content.strip()
            
            # Execute the Cypher query
            with self.driver.session() as session:
                result = session.run(cypher_query, limit=limit)
                records = [record.data() for record in result]
                
            return {
                "query": query,
                "cypher": cypher_query,
                "results": records
            }
            
        except Exception as e:
            raise ValueError(f"Error querying knowledge graph: {str(e)}")

# FastAPI Application
app = FastAPI(
    title="Knowledge Graph Builder API",
    description="API for building knowledge graphs from educational materials",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the knowledge graph builder
kg_builder = NCERTKnowledgeGraphBuilder()

# In-memory job storage (in production, use a database)
processing_jobs = {}

def update_job_status(job_id: str, status: str, message: str, progress: float = None, 
                     total_chunks: int = None, processed_chunks: int = None):
    """Update the status of a processing job"""
    processing_jobs[job_id] = {
        "job_id": job_id,
        "status": status,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "progress": progress,
        "total_chunks": total_chunks,
        "processed_chunks": processed_chunks
    }

@app.post("/upload", response_model=ProcessingStatus)
async def upload_book(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    book_reference: str = Form(...),
    grade_level: Optional[str] = Form(None),
    subject: Optional[str] = Form(None)
):
    """Upload a PDF book for knowledge graph creation"""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Generate a unique job ID
    job_id = str(uuid.uuid4())
    
    # Save the uploaded file
    file_path = os.path.join(kg_builder.settings.UPLOAD_DIR, f"{job_id}_{file.filename}")
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
    # Create context for processing
    context = {
        "book_reference": book_reference,
        "grade_level": grade_level,
        "subject": subject
    }
    
    # Initialize job status
    update_job_status(job_id, "queued", "Job queued for processing")
    
    # Process the book in the background
    background_tasks.add_task(
        process_book_task, 
        job_id=job_id, 
        file_path=file_path, 
        context=context
    )
    
    return ProcessingStatus(
        job_id=job_id,
        status="queued",
        message="Book uploaded and queued for processing",
        timestamp=datetime.now().isoformat()
    )

async def process_book_task(job_id: str, file_path: str, context: Dict):
    """Background task to process a book"""
    try:
        update_job_status(job_id, "processing", "Started processing book")
        
        # Process the book
        kg_builder.load_and_process_pdf(
            file_path=file_path, 
            context=context,
            job_id=job_id,
            status_callback=update_job_status
        )
        
        # Update final status
        update_job_status(job_id, "completed", "Book processing completed", 100.0)
        
    except Exception as e:
        update_job_status(job_id, "failed", f"Processing failed: {str(e)}")
        
    finally:
        # Clean up the uploaded file (optional)
        # os.remove(file_path)
        pass

@app.get("/status/{job_id}", response_model=ProcessingStatus)
async def get_job_status(job_id: str):
    """Get the status of a processing job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return ProcessingStatus(**processing_jobs[job_id])

@app.post("/query", response_model=Dict)
async def query_graph(query_request: QueryRequest):
    """Query the knowledge graph using natural language"""
    try:
        result = kg_builder.query_knowledge_graph(
            query=query_request.query,
            limit=query_request.limit
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying graph: {str(e)}")

@app.get("/entities", response_model=List[Dict])
async def get_entities(
    type: Optional[str] = Query(None, description="Filter by entity type"),
    limit: int = Query(100, description="Maximum number of entities to return")
):
    """Get entities from the knowledge graph"""
    try:
        with kg_builder.driver.session() as session:
            if type:
                query = f"MATCH (n:`{type}`) RETURN n LIMIT $limit"
            else:
                query = "MATCH (n) RETURN n LIMIT $limit"
            
            result = session.run(query, limit=limit)
            entities = [
                {
                    "id": record["n"].id,
                    "name": record["n"].get("name", ""),
                    "type": list(record["n"].labels)[0],
                    "properties": dict(record["n"])
                }
                for record in result
            ]
            
        return entities
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching entities: {str(e)}")

@app.get("/relationships", response_model=List[Dict])
async def get_relationships(
    type: Optional[str] = Query(None, description="Filter by relationship type"),
    limit: int = Query(100, description="Maximum number of relationships to return")
):
    """Get relationships from the knowledge graph"""
    try:
        with kg_builder.driver.session() as session:
            if type:
                query = f"MATCH (a)-[r:`{type}`]->(b) RETURN a, r, b LIMIT $limit"
            else:
                query = "MATCH (a)-[r]->(b) RETURN a, r, b LIMIT $limit"
            
            result = session.run(query, limit=limit)
            relationships = [
                {
                    "id": record["r"].id,
                    "type": type(record["r"]).__name__,
                    "source": {
                        "id": record["a"].id,
                        "name": record["a"].get("name", ""),
                        "type": list(record["a"].labels)[0]
                    },
                    "target": {
                        "id": record["b"].id,
                        "name": record["b"].get("name", ""),
                        "type": list(record["b"].labels)[0]
                    },
                    "properties": dict(record["r"])
                }
                for record in result
            ]
            
        return relationships
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching relationships: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run("knowledge_graph:app", host="0.0.0.0", port=8000, reload=True)
