# Stage 2: Context Selection Prompts

VK_USER_PROMPT = """Please combine detailed captions, detected targets (object), and recognized text (OCR) from the same image to generate a more specific caption that is relevant to answering the question.
Question: {question}
Image's captions: {captions}
Object: {objects}
OCR: {ocrs}
Visual Knowledge: """

EK_USER_PROMPT = """Summarize the retrieved knowledge snippets relevant to this image caption and question.
Caption: {caption}
Question: {question}
Knowledge snippets: {knowledge}
Summarized Knowledge: """

IK_USER_PROMPT = """Based on the image caption and question, extract implicit knowledge that could help answer the question. This knowledge should come from your pretrained knowledge and be relevant to the visual content.
Caption: {caption}
Question: {question}
Implicit Knowledge: """

# Stage 3: Context Compression Prompts

ENTITY_EXTRACTION = """Given a sentence, possible entities may include: [person,organizations,locations,everyday objects,animals and plants,geography,occupations,food,art and culture,technology,composite entities,time expressions,quantities,monetary values,abstract entities], and almost all subjects and objects would qualify. Find all concrete entities based on the provided sentence.
Sentence: {sentence}
Entities:"""

RELATION_EXTRACTION = """Given a sentence, and all entities within the sentence (some entities may not be in the list and some entities may be duplicated). 
Extract all relationships between entities which directly stated in the sentence. Relationships that involve uncertainty, such as those expressed by words like "possible" or "maybe" should not be present.
Every relationship stated as a triple: (E_A, E_B, Relation).
Sentence: {sentence}
Entities: {entities}
Relation: """

KEY_ENTITY_EXTRACTION = """Please extract all Key Entities from the Entity List. Key entities refer to those explicitly or implicitly mentioned in the question that play a crucial role in answering it, and there may be 1-3 such entities.
Sentence: {sentence}
Entity List: {entities}
Question: {question}
Key Entities: """

RELATION_REFINEMENT = """Please extract the relational triples where the subject entity and object entity of the triples are completely irrelevant to any key entity from the following set of all relational triples (which constitutes a knowledge graph). Specifically, we need to extract the triples (E_A, Relation, E_B) that satisfy the condition: neither E_A nor E_B can reach any key entity (or entities that are semantically equivalent to the key entity) via the paths in the relational graph.
Original Relations: {relations}
Key Entity List: {key_entities}
output example:[(irrelevant_E_A, Relation_AB, irrelevant_E_B),...]
Irrelevant Relations: """
