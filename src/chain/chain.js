import dotenv from 'dotenv';
import { MistralAI } from "@langchain/mistralai";
import path from 'path';
import { fileURLToPath } from 'url';
import { formatPrompt } from '../prompt/prompt.js';
import { preprocessQuery, normalizeTextForEmbedding, vectorSearch, bm25Search, hybridSearch } from '../scripts/preprocessquery.js';
import { ChatGroq } from "@langchain/groq"

// Get directory name in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load environment variables from the root .env file
dotenv.config({ path: path.resolve(__dirname, '../../.env') });

async function getHybridSearchResults(query, topK = 10) {
    console.log('🔍 Performing hybrid search...');
    const preprocessed = preprocessQuery(query);
    const queryVector = await normalizeTextForEmbedding(preprocessed.normalized);
    const vectorResults = await vectorSearch(queryVector, topK);
    const bm25Results = await bm25Search(query, topK);
    const results = hybridSearch(vectorResults, bm25Results, topK);
    return results;
}

function formatDocumentContext(searchResults) {
    // Format search results into a readable document context
    return searchResults.map((result, index) => `
Story ${index + 1}:
Key: ${result.key}
Summary: ${result.summary}
Status: ${result.status?.name || 'Unknown'}
Score: ${result.hybridScore.toFixed(4)}
Found in: ${result.foundIn}
---`).join('\n');
}

async function main() {
    // mistralai();
    await groq();
}

async function mistralai() {
    console.log('🚀 Starting User Story Validation Chain...');
    console.log('📋 API Key loaded:', process.env.MISTRAL_API_KEY ? 'Yes' : 'No');

    // Step 1: Define the new user story to validate
    const newStory = `
    Key: HC-NEW-001
    Summary: Nurse Activities Raise Request
    Description: As a user, I want to nurse activities raise request so that the related functionality works as expected.
    Status: Draft
    Priority: P1
    `;

    // Step 2: Get hybrid search results for context
    const searchQuery = "nurse activities raise request";
    const searchResults = await getHybridSearchResults(newStory, 5);

    console.log(`\n📊 Retrieved ${searchResults.length} related stories for context`);

    // Step 3: Format document context from search results
    const documentContext = formatDocumentContext(searchResults);

    // Step 4: Create the prompt using formatPrompt
    const prompt = formatPrompt(documentContext, newStory);

    console.log('\n📝 Generated Prompt Length:', prompt.length, 'characters');
    console.log('\n' + '='.repeat(80));
    console.log('PROMPT TO BE SENT:');
    console.log('='.repeat(80));
    console.log(prompt); // Preview first 500 chars

    // Step 5: Initialize Mistral AI
    const llm = new MistralAI({
        model: "codestral-latest",
        temperature: 0,
        maxTokens: 2000, // Increased for detailed response
        maxRetries: 2,
        apiKey: process.env.MISTRAL_API_KEY,
        timeout: 30000
    });

    console.log('📤 Sending request to Mistral AI...');

    try {
        const response = await llm.invoke(prompt);
        console.log('\n✅ Response received:');
        console.log('='.repeat(80));
        console.log(response);
        console.log('='.repeat(80));
    } catch (error) {
        console.error('❌ Error:', error.message);
        console.error('Full error:', error);
    } finally {
        console.log('\n✨ Done!');
        process.exit(0);
    }
}

async function groq() {
    console.log('🚀 Starting Groq test...');
    console.log('📋 API Key loaded:', process.env.GROQ_API_KEY ? 'Yes' : 'No');

    // Step 1: Define the new user story to validate
    const newStory = `
    Key: HC-NEW-001
    Summary: Nurse Activities Raise Request
    Description: As a user, I want to nurse activities raise request so that the related functionality works as expected.
    Status: Draft
    Priority: P1
    `;

    // Step 2: Get hybrid search results for context
    const searchQuery = "nurse activities raise request";
    const searchResults = await getHybridSearchResults(newStory, 5);

    console.log(`\n📊 Retrieved ${searchResults.length} related stories for context`);

    // Step 3: Format document context from search results
    const documentContext = formatDocumentContext(searchResults);

    // Step 4: Create the prompt using formatPrompt
    const prompt = formatPrompt(documentContext, newStory);

    console.log('\n📝 Generated Prompt Length:', prompt.length, 'characters');
    console.log('\n' + '='.repeat(80));
    console.log('PROMPT TO BE SENT:');
    console.log('='.repeat(80));
    console.log(prompt); // Preview first 500 chars

    try {
        const llm = new ChatGroq({
            model: "openai/gpt-oss-120b",
            temperature: 0,
            maxTokens: undefined,
            maxRetries: 2,
            apiKey: process.env.GROQ_API_KEY,
            timeout: 30000, // 30 second timeout
        });

        console.log('📤 Sending request to Groq...');

        const aiMsg = await llm.invoke([
            // {
            //     role: "system",
            //     content: "You are a helpful assistant that translates English to French. Translate the user sentence.",
            // },
            { role: "user", content: prompt },
        ]);

        console.log('\n✅ Response received:');
        console.log('='.repeat(80));
        console.log(aiMsg);
        console.log('='.repeat(80));
    } catch (error) {
        console.error('❌ Error:', error.message);
        console.error('Full error:', error);
    } finally {
        console.log('✨ Done!');
        process.exit(0);
    }
}

// async function main() {
//     console.log('🚀 Starting Mistral AI test...');
//     console.log('📋 API Key loaded:', process.env.MISTRAL_API_KEY ? 'Yes' : 'No');

//     const llm = new MistralAI({
//         model: "codestral-latest",
//         temperature: 0,
//         maxTokens: 100,
//         maxRetries: 2,
//         apiKey: process.env.MISTRAL_API_KEY || "dV19xYSU5sNZh0GAUFXqsCX7OuFpqnba",
//         timeout: 30000 // 30 second timeout
//     });

//     const inputText1 = "MistralAI is an AI company that ";

//     const inputText2 = `
// ### INSTRUCTION
// You are an expert Agile QA and story reviewer. Your job is to:
// 1. Evaluate the quality of the provided user story based on specific criteria.
// 2. Provide a refined version of the same story with better clarity, structure, and grammar.
// 3. Use only information explicitly available in the document. 
// 4. If information is missing, respond with: "I cannot find this information in the provided document."
// 5. Assign a 1–5 score for each criterion:
//    - Format: Does it follow "As a [role], I want [goal], so that [benefit]"? (1–5)
//    - Clarity: Is the intent and outcome clear and unambiguous? (1–5)
//    - Testability: Are acceptance criteria measurable/testable? (1–5)
//    - Completeness: Does it contain all essential story components? (1–5)
//    - Consistency: Does it align with patterns from the RAG document? (1–5)
//    - Grammar: Is it grammatically correct and well-phrased? (1–5)
// 6. The overall quality score = average of the above six scores.

// ---

// ### CONTEXT
// You will be provided with:
// 1. A document containing existing user stories (RAG data)
// 2. A new user story to validate and refine

// Document:
// ---

// Story 1:
// Key: HC-123
// Summary: Nurse Activities Raise Request
// Status: Done
// Score: 0.9183
// Found in: both
// ---

// Story 2:
// Key: HC-84
// Summary: Daycare Nurse Activities Raise
// Status: In Progress
// Score: 0.7855
// Found in: both
// ---

// Story 3:
// Key: HC-229
// Summary: Doctor Activities Raise Request
// Status: Done
// Score: 0.7832
// Found in: both
// ---

// Story 4:
// Key: HC-213
// Summary: Nurse Activities Prescription View
// Status: To Do
// Score: 0.3778
// Found in: vector
// ---

// Story 5:
// Key: HC-173
// Summary: Nurse Activities Daily Assessment
// Status: Done
// Score: 0.3773
// Found in: vector
// ---

// Story 6:
// Key: HC-220
// Summary: Nurse Activities Nurse Notes
// Status: To Do
// Score: 0.2725
// Found in: bm25
// ---

// Story 7:
// Key: HC-118
// Summary: Consultant Dashboard Raise Request
// Status: In Progress
// Score: 0.2711
// Found in: bm25
// ---
// ---

// New Story:
// ---

//     Key: HC-NEW-001
//     Summary: Nurse Activities Raise Request
//     Description: As a user, I want to nurse activities raise request so that the related functionality works as expected.
//     Status: Draft
//     Priority: P1

// ---

// ---

// ### EXAMPLES

// Example 1:
// Input Story:
// "As a user, I want to nurse activities raise request so that the related functionality works as expected."

// Validation:
// - Format: 3 (follows partial structure)
// - Clarity: 2 (unclear role/goal)
// - Testability: 3 (implied but vague)
// - Completeness: 3
// - Consistency: 4 (similar to HC-84 and HC-123)
// - Grammar: 2
// Overall Score: 2.83/5

// Issues:
// - Role not specific ("user" instead of "nurse")
// - Goal unclear ("raise request" is ambiguous)
// - Benefit vague ("related functionality works as expected")

// Refined Story:
// "As a nurse, I want to raise and manage patient activity requests so that I can track and update their progress efficiently."

// Acceptance Criteria:
// - Given a nurse with valid access, when they open the activity dashboard, then they should be able to create, update, and view patient activity requests successfully.
// - When changes are saved, they should reflect instantly without delay.

// Citations:
// Based on similar structure and clarity found in HC-84 and HC-123 from the provided document.

// ---

// ### PERSONA
// You are a Senior Agile Quality Analyst specializing in evaluating and refining user stories for clarity, completeness, and testability. You ensure alignment with best practices and consistency across RAG-based stories.

// ---

// ### OUTPUT FORMAT
// Provide the result in this exact structure:
// 1. **Story Quality Validation**
//    - Format: x/5
//    - Clarity: x/5
//    - Testability: x/5
//    - Completeness: x/5
//    - Consistency: x/5
//    - Grammar: x/5
//    - **Overall Quality Score:** x.xx/5
//    - Issues Identified: [List key issues briefly]

// 2. **Refined Story**
//    - Summary: [Improved summary]
//    - Description: [Rewritten user story]
//    - Acceptance Criteria: [Refined Given-When-Then statements]

// 3. **Citations**
//    - Referenced stories (e.g., HC-84, HC-173)
//    - Direct quotes or matching phrasing examples if applicable

// ---

// ### TONE
// Professional, objective, and audit-style. Be concise and precise. Avoid assumptions or external knowledge. Only validate and refine based on the document provided.`;

//     console.log('📤 Sending request to Mistral AI...');
//     console.log('💬 Input:', inputText1);

//     try {
//         const response = await llm.invoke(inputText1);
//         console.log('✅ Response received:');
//         console.log(response);
//     } catch (error) {
//         console.error('❌ Error:', error.message);
//         console.error('Full error:', error);
//     } finally {
//         console.log('✨ Done!');
//         process.exit(0); // Force exit
//     }
// };

main();