import dotenv from "dotenv";
import axios from "axios";
import { MongoClient } from 'mongodb';

dotenv.config();

const TESTLEAF_API_BASE = process.env.TESTLEAF_API_BASE;
const USER_EMAIL = process.env.TESTLEAF_USER_EMAIL;
const AUTH_TOKEN = process.env.TESTLEAF_AUTH_TOKEN;

export { preprocessQuery, normalizeTextForEmbedding, vectorSearch, bm25Search, hybridSearch };

async function main() {
    const workingQuery = "HC-257"
    const preprocessed = preprocessQuery(workingQuery);
    console.log(preprocessed);
    const queryVector = await normalizeTextForEmbedding(preprocessed.normalized);
    const vectorResults = await vectorSearch(queryVector, 10);
    const bm25Results = await bm25Search(workingQuery, 10);
    const results = hybridSearch(vectorResults, bm25Results, 10);
    return results;
}
function preprocessQuery(rawQuery) {
    const startTime = Date.now();
    const normalizeResult = normalizeComplete(rawQuery, {
        lowercase: true,
        preserveHyphens: true,
        preserveNumbers: true
    });
    const endTime = Date.now();

    console.log(`🕒 Query Preprocessing Time: ${endTime - startTime} ms`);
    console.log(`🔤 Original Query: "${rawQuery}"`);
    console.log(`📝 Normalized Query: "${normalizeResult.normalized}"`);
    return normalizeResult;
}


function normalizeComplete(text, options = {}) {
    if (!text) {
        return {
            normalized: '',
            metadata: {
                original: '',
                testCaseIds: [],
                tokens: []
            }
        };
    }

    // Apply standard normalization
    const normalized = normalize(text, options)

    return {
        normalized,
        metadata: {
            original: text
        }
    };
}


function normalize(text, options = {}) {
    const {
        lowercase = true,
        trimWhitespace = true,
        removeExtraSpaces = true,
        removeSpecialChars = false,
        preserveHyphens = true,
        preserveUnderscores = false,
        preserveNumbers = true
    } = options;

    if (!text || typeof text !== 'string') {
        return '';
    }

    let normalized = text;

    // Trim whitespace
    if (trimWhitespace) {
        normalized = normalized.trim();
    }

    // Remove extra spaces (multiple spaces → single space)
    if (removeExtraSpaces) {
        normalized = normalized.replace(/\s+/g, ' ');
    }

    // Convert to lowercase
    if (lowercase) {
        normalized = normalized.toLowerCase();
    }

    // Remove special characters
    if (removeSpecialChars) {
        let pattern = '[^a-zA-Z0-9\\s';
        if (preserveHyphens) pattern += '-';
        if (preserveUnderscores) pattern += '_';
        pattern += ']';

        const regex = new RegExp(pattern, 'g');
        normalized = normalized.replace(regex, ' ');

        // Clean up extra spaces created by removal
        normalized = normalized.replace(/\s+/g, ' ').trim();
    }

    // Handle numbers
    if (!preserveNumbers) {
        normalized = normalized.replace(/\d+/g, '');
        normalized = normalized.replace(/\s+/g, ' ').trim();
    }

    return normalized;
}

async function normalizeTextForEmbedding(text) {
    const embeddingResponse = await axios.post(
        `${TESTLEAF_API_BASE}/embedding/text/${USER_EMAIL}`,
        {
            input: text,
            model: "text-embedding-3-small"
        },
        {
            headers: {
                'Content-Type': 'application/json',
                ...(AUTH_TOKEN && { 'Authorization': `Bearer ${AUTH_TOKEN}` })
            }
        }
    );
    console.log(`📡 Server Response Received:`);
    console.log(`   📊 Status: ${embeddingResponse.status}`);
    console.log(`   📋 Response Status: ${embeddingResponse.data.status}`);
    console.log(`   💬 Message: ${embeddingResponse.data.message || 'No message'}`);
    return embeddingResponse;
}

async function vectorSearch(queryVector, topK = 5, filters = {}) {

    const mongoClient = new MongoClient(process.env.MONGODB_URI, {
        ssl: true,
        tlsAllowInvalidCertificates: true,
        tlsAllowInvalidHostnames: true,
        serverSelectionTimeoutMS: 30000,
        connectTimeoutMS: 30000,
        socketTimeoutMS: 30000,
    });
    await mongoClient.connect();


    const db = mongoClient.db(process.env.DB_NAME);
    const collection = db.collection(process.env.USER_STORIES_COLLECTION);
    const embedding = queryVector.data.data[0].embedding;
    console.log(embedding);
    // Calculate candidates and internal limit for vector search
    const requestedLimit = parseInt(topK);
    const numCandidates = Math.max(100, requestedLimit * 10); // At least 100 candidates
    const vectorSearchLimit = Math.min(numCandidates, requestedLimit * 10); // Limit must be <= numCandidates

    // Build vector search WITHOUT pre-filtering (to avoid index requirement)
    const vectorSearchStage = {
        $vectorSearch: {
            queryVector: embedding,
            path: "embedding",
            numCandidates: numCandidates,
            limit: vectorSearchLimit, // Get more candidates for post-filtering
            index: process.env.VECTOR_INDEX_NAME
        }
    };

    // Build the pipeline
    const pipeline = [
        vectorSearchStage,
        {
            $addFields: {
                score: { $meta: "vectorSearchScore" }
            }
        }
    ];

    // Apply metadata filters using $match stage (works without index)
    if (Object.keys(filters).length > 0) {
        const matchConditions = {};
        Object.entries(filters).forEach(([key, value]) => {
            if (value) {
                matchConditions[key] = value;
            }
        });
        pipeline.push({
            $match: matchConditions
        });
        console.log('🔍 Applying filters with $match:', matchConditions);
    }

    // Add limit after filtering
    pipeline.push({
        $limit: requestedLimit
    });

    // Project desired fields
    pipeline.push({
        $project: {
            _id: 1,
            key: 1,
            summary: 1,
            status: 1,
            score: 1
        }
    });

    console.log('🔍 Filters:', JSON.stringify(filters));
    console.log('🔍 Pipeline:', JSON.stringify(pipeline, null, 2));

    const results = await collection.aggregate(pipeline).toArray();
    console.log('✅ Found results:', results.length);
    results.forEach((result, index) => {
        console.log(`------------------------------`);
        // print result details
        console.log(`   ${index + 1}. [Score: ${result.score.toFixed(4)}] Key: ${result.key}, Summary: ${result.summary.substring(0, 50)}...`);
    });
    return results;
}

async function bm25Search(queryText, topK = 5, filters = {}, fields = ["key", "summary", "description"]) {
    // Placeholder for BM25 search implementation
    console.log(`🔍 Performing BM25 search for: "${queryText}" with topK=${topK} and filters=${JSON.stringify(filters)}`);
    const mongoClient = new MongoClient(process.env.MONGODB_URI, {
        ssl: true,
        tlsAllowInvalidCertificates: true,
        tlsAllowInvalidHostnames: true,
        serverSelectionTimeoutMS: 30000,
        connectTimeoutMS: 30000,
        socketTimeoutMS: 30000,
    });
    await mongoClient.connect();
    if (!queryText || queryText.trim() === '') {
        return res.status(400).json({ error: 'Query is required' });
    }

    const db = mongoClient.db(process.env.DB_NAME);
    const collection = db.collection(process.env.USER_STORIES_COLLECTION);

    // Build BM25 search pipeline
    const pipeline = [
        {
            $search: {
                index: process.env.BM25_INDEX_NAME,
                text: {
                    query: queryText,
                    path: fields,
                    fuzzy: {
                        maxEdits: 1,
                        prefixLength: 2
                    }
                }
            }
        },
        {
            $addFields: {
                score: { $meta: "searchScore" }
            }
        }
    ];

    // Apply filters if provided
    if (Object.keys(filters).length > 0) {
        const matchConditions = {};
        Object.entries(filters).forEach(([key, value]) => {
            if (value && value !== '') {
                matchConditions[key] = value;
            }
        });

        if (Object.keys(matchConditions).length > 0) {
            pipeline.push({ $match: matchConditions });
        }
    }

    //Project desired fields and limit results
    pipeline.push(
        {
            $project: {
                _id: 1,
                key: 1,
                summary: 1,
                status: 1,
                score: 1
            }
        },
        {
            $limit: parseInt(topK)
        }
    );
    console.log('🔍 BM25 Pipeline:', JSON.stringify(pipeline, null, 2));

    const results = await collection.aggregate(pipeline).toArray();

    await mongoClient.close();
    console.log('✅ Found results:', results.length);
    results.forEach((result, index) => {
        console.log(`------------------------------`);
        // print result details
        console.log(`   ${index + 1}. [Score: ${result.score.toFixed(4)}] Key: ${result.key}, Summary: ${result.summary.substring(0, 50)}...`);
    });
    return results;
}

function hybridSearch(vectorResults, bm25Results, topK = 5, filters = {}, vectorWeight = 0.5, bm25Weight = 0.5) {
    // 3. Normalize and combine scores
    console.log('🔀 Combining results...');

    // Normalize BM25 scores
    const bm25Scores = bm25Results.map(r => r.score);
    const bm25Max = Math.max(...bm25Scores, 1);
    const bm25Min = Math.min(...bm25Scores, 0);
    const bm25Range = bm25Max - bm25Min || 1;

    // Normalize Vector scores
    const vectorScores = vectorResults.map(r => r.score);
    const vectorMax = Math.max(...vectorScores, 1);
    const vectorMin = Math.min(...vectorScores, 0);
    const vectorRange = vectorMax - vectorMin || 1;

    const resultMap = new Map();

    // Add BM25 results with normalized scores
    bm25Results.forEach(result => {
        const key = result._id.toString();
        const normalizedScore = (result.score - bm25Min) / bm25Range;
        resultMap.set(key, {
            ...result,
            bm25ScoreNormalized: normalizedScore,
            vectorScore: 0,
            vectorScoreNormalized: 0,
            hybridScore: normalizedScore * bm25Weight,
            foundIn: 'bm25'
        });
    });

    // Add/merge vector results with normalized scores
    vectorResults.forEach(result => {
        const key = result._id.toString();
        const normalizedScore = (result.score - vectorMin) / vectorRange;

        if (resultMap.has(key)) {
            // Merge - found in both
            const existing = resultMap.get(key);
            existing.vectorScore = result.score;
            existing.vectorScoreNormalized = normalizedScore;
            existing.hybridScore += normalizedScore * vectorWeight;
            existing.foundIn = 'both';
        } else {
            // New result - only in vector
            resultMap.set(key, {
                ...result,
                bm25Score: 0,
                bm25ScoreNormalized: 0,
                vectorScoreNormalized: normalizedScore,
                hybridScore: normalizedScore * vectorWeight,
                foundIn: 'vector'
            });
        }
    });

    let combinedResults = Array.from(resultMap.values());
    combinedResults.sort((a, b) => b.hybridScore - a.hybridScore);

    // Limit results
    const finalResults = combinedResults.slice(0, parseInt(10));

    // Calculate statistics
    const bothCount = finalResults.filter(r => r.foundIn === 'both').length;
    const bm25OnlyCount = finalResults.filter(r => r.foundIn === 'bm25').length;
    const vectorOnlyCount = finalResults.filter(r => r.foundIn === 'vector').length;

    console.log(`🔍 Results Breakdown: Both=${bothCount}, BM25 Only=${bm25OnlyCount}, Vector Only=${vectorOnlyCount}`);

    // Print final results
    console.log(finalResults)
    return finalResults;

}

// main();
