import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';
import { ChatGroq } from "@langchain/groq";
import { formatPrompt } from '../prompt/prompt.js';
import { preprocessQuery, normalizeTextForEmbedding, vectorSearch, bm25Search, hybridSearch } from '../scripts/preprocessquery.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config({ path: path.resolve(__dirname, '../../.env') });

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../public')));

// Helper Functions
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
    return searchResults.map((result, index) => `
Story ${index + 1}:
Key: ${result.key}
Summary: ${result.summary}
Status: ${result.status?.name || 'Unknown'}
Score: ${result.hybridScore.toFixed(4)}
Found in: ${result.foundIn}
---`).join('\n');
}

// API Routes
app.post('/api/validate-story', async (req, res) => {
    try {
        const { userStory, searchQuery } = req.body;

        if (!userStory) {
            return res.status(400).json({
                error: 'User story is required'
            });
        }

        console.log('📝 Received user story for validation');

        // Step 1: Get hybrid search results
        const searchResults = await getHybridSearchResults(
            searchQuery || userStory,
            5
        );

        console.log(`📊 Retrieved ${searchResults.length} related stories`);

        // Step 2: Format document context
        const documentContext = formatDocumentContext(searchResults);

        // Step 3: Create prompt
        const prompt = formatPrompt(documentContext, userStory);

        console.log('📤 Sending to Groq...');

        // Step 4: Call Groq
        const llm = new ChatGroq({
            model: "llama-3.3-70b-versatile",
            temperature: 0,
            maxTokens: 2000,
            apiKey: process.env.GROQ_API_KEY,
            timeout: 30000,
        });

        const response = await llm.invoke([
            { role: "user", content: prompt }
        ]);

        console.log('✅ Response received from Groq');

        // Step 5: Return response
        res.json({
            success: true,
            validation: response.content,
            relatedStories: searchResults,
            documentContext: documentContext
        });

    } catch (error) {
        console.error('❌ Error:', error.message);
        res.status(500).json({
            error: error.message,
            details: 'Failed to validate user story'
        });
    }
});

// Health check endpoint
app.get('/api/health', (req, res) => {
    res.json({
        status: 'ok',
        message: 'Server is running',
        groqKeyLoaded: !!process.env.GROQ_API_KEY
    });
});

app.listen(PORT, () => {
    console.log(`🚀 Server running on http://localhost:${PORT}`);
    console.log(`📋 Groq API Key loaded: ${process.env.GROQ_API_KEY ? 'Yes' : 'No'}`);
});