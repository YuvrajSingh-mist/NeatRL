#!/usr/bin/env node

/**
 * Secure SmolHub Data Generator
 * This script fetches SmolHub playground data from GitHub using your personal access token
 * and generates the _data/smolhub_playground.json file for Jekyll to use.
 * 
 * Usage: GITHUB_TOKEN=your_token_here node generate-smolhub-data.js
 */

const fs = require('fs');
const path = require('path');

// Use built-in fetch for Node.js 18+ or fallback to node-fetch
let fetch;
try {
    fetch = globalThis.fetch;
    if (!fetch) {
        fetch = require('node-fetch');
    }
} catch (e) {
    try {
        fetch = require('node-fetch');
    } catch (e2) {
        console.error('❌ Error: fetch not available. Please use Node.js 18+ or install node-fetch');
        process.exit(1);
    }
}

// Check for GitHub token (optional for public repos)
const GITHUB_TOKEN = process.env.GITHUB_TOKEN;
if (!GITHUB_TOKEN) {
    console.warn('⚠️ No GITHUB_TOKEN provided - using public API (rate limited)');
}

// Configuration
const REPO_OWNER = 'YuvrajSingh-mist';
const REPO_NAME = 'SmolHub';
const REPO_BRANCH = 'main';
const OUTPUT_FILE = path.join(__dirname, '_data', 'smolhub_playground.json');
// Exclude non-project directories
const EXCLUDED_DIRS = ['hf-spaces'];

console.log('🚀 Starting SmolHub playground data generation...');
console.log(`📂 Repository: ${REPO_OWNER}/${REPO_NAME}`);
console.log(`🌿 Branch: ${REPO_BRANCH}`);
console.log(`📄 Output file: ${OUTPUT_FILE}`);

async function fetchWithAuth(url) {
    const headers = {
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'Portfolio-SmolHub-Generator'
    };
    
    // Add authorization header if token is available
    if (GITHUB_TOKEN) {
        headers['Authorization'] = `token ${GITHUB_TOKEN}`;
    }
    
    const response = await fetch(url, { headers });
    
    if (!response.ok) {
        throw new Error(`GitHub API error: ${response.status} ${response.statusText}`);
    }
    
    return response.json();
}

async function fetchRepositoryContents() {
    console.log('📡 Fetching repository contents...');
    const url = `https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/contents/?ref=${REPO_BRANCH}`;
    const contents = await fetchWithAuth(url);
    
    // Filter only directories and exclude known non-project folders
    const directories = contents
        .filter(item => item.type === 'dir')
        .filter(item => !EXCLUDED_DIRS.includes(item.name));
    console.log(`📁 Found ${directories.length} playground directories`);
    
    return directories;
}

async function fetchReadmeContent(folderName) {
    try {
        console.log(`📖 Fetching README for ${folderName}...`);
        const readmeUrl = `https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/contents/${folderName}/README.md?ref=${REPO_BRANCH}`;
        const readmeResponse = await fetchWithAuth(readmeUrl);
        
        // Decode base64 content
        const readmeContent = Buffer.from(readmeResponse.content, 'base64').toString('utf8');
        
        // Extract description from README (first paragraph or summary)
        const lines = readmeContent.split('\n');
        let description = `SmolHub playground project: ${folderName}`;
        
        // Look for description in README
        for (let i = 0; i < lines.length; i++) {
            let line = lines[i].trim();
            if (line && !line.startsWith('#') && !line.startsWith('![') && line.length > 10) {
                // Strip leading list markers and excessive whitespace for clean overview
                line = line.replace(/^[\-\*\+\u2022\u2013\u2014]\s+/, '');
                line = line.replace(/\s+/g, ' ').trim();
                // Use the full first descriptive line as description (no hard truncation)
                description = line;
                break;
            }
        }
        
        return {
            description,
            readmeContent
        };
        
    } catch (error) {
        console.warn(`⚠️ Could not fetch README for ${folderName}:`, error.message);
        return {
            description: `SmolHub playground project: ${folderName}`,
            readmeContent: `# ${folderName}\n\nThis project is part of the SmolHub Playground collection.\n\n[View on GitHub](https://github.com/${REPO_OWNER}/${REPO_NAME}/tree/${REPO_BRANCH}/${folderName})`
        };
    }
}

function formatDisplayName(folderName) {
    // Convert folder names to proper display names
    return folderName
        .replace(/([A-Z])/g, ' $1') // Add space before capital letters
        .replace(/^./, str => str.toUpperCase()) // Capitalize first letter
        .trim();
}

function generateProjectTags(name, description, readmeContent) {
    const content = (name + ' ' + description + ' ' + (readmeContent || '')).toLowerCase();
    const tags = [];
    
    console.log(`🏷️ Auto-tagging ${name}: Analyzing content...`);
    
    // Specific model architecture tags
    if (content.includes('mixtral') || content.includes('mixture of experts') || content.includes('moe')) {
        tags.push('mixtral');
        console.log(`  ✅ Added Mixtral tag`);
    }
    
    if (content.includes('llama') || content.includes('language model')) {
        tags.push('llama');
        console.log(`  ✅ Added Llama tag`);
    }
    
    if (content.includes('kimi') || content.includes('long context') || content.includes('deepseek')) {
        tags.push('kimi');
        console.log(`  ✅ Added Kimi tag`);
    }
    
    // Architecture and framework tags
    if (content.includes('transformer') || content.includes('attention') || content.includes('encoder') || content.includes('decoder')) {
        tags.push('transformer');
        console.log(`  ✅ Added Transformer tag`);
    }
    
    if (content.includes('pytorch') || content.includes('torch')) {
        tags.push('pytorch');
        console.log(`  ✅ Added PyTorch tag`);
    }
    
    // Task-specific tags
    if (content.includes('translation') || content.includes('english to hindi') || content.includes('hindi') || content.includes('samanantar')) {
        tags.push('translation');
        console.log(`  ✅ Added Translation tag`);
    }
    
    if (content.includes('story') && (content.includes('generation') || content.includes('tinystories') || name.toLowerCase().includes('story'))) {
        tags.push('storytelling');
        console.log(`  ✅ Added Storytelling tag`);
    }
    
    if (content.includes('text generation') && !content.includes('translation')) {
        tags.push('generation');
        console.log(`  ✅ Added Generation tag`);
    }
    
    // Size and complexity tags
    if (content.includes('smol') || content.includes('small') || content.includes('mini') || content.includes('tiny') || content.includes('compact')) {
        tags.push('compact');
        console.log(`  ✅ Added Compact tag`);
    }
    
    // Domain tags
    if (content.includes('nlp') || content.includes('natural language') || content.includes('language processing')) {
        tags.push('nlp');
        console.log(`  ✅ Added NLP tag`);
    }
    
    if (content.includes('vision') || content.includes('image') || content.includes('visual') || content.includes('cv')) {
        tags.push('vision');
        console.log(`  ✅ Added Vision tag`);
    }
    
    // Training and optimization tags
    if (content.includes('distributed') || content.includes('ddp') || content.includes('multi-gpu')) {
        tags.push('distributed');
        console.log(`  ✅ Added Distributed tag`);
    }
    
    if (content.includes('gradio') || content.includes('web interface') || content.includes('interactive')) {
        tags.push('interactive');
        console.log(`  ✅ Added Interactive tag`);
    }
    
    if (content.includes('flash attention') || content.includes('optimization') || content.includes('liger')) {
        tags.push('optimized');
        console.log(`  ✅ Added Optimized tag`);
    }
    
    // Purpose tags
    if (content.includes('experimental') || content.includes('playground') || content.includes('proof')) {
        tags.push('experimental');
        console.log(`  ✅ Added Experimental tag`);
    }
    
    if (content.includes('education') || content.includes('learning') || content.includes('tutorial') || content.includes('teaching')) {
        tags.push('educational');
        console.log(`  ✅ Added Educational tag`);
    }
    
    // If no specific tags found, add default playground tag
    if (tags.length === 0) {
        tags.push('playground');
        console.log(`  ⚠️ No specific tags found, added default playground tag`);
    }
    
    console.log(`🏷️ Final tags for ${name}: [${tags.join(', ')}]`);
    return tags;
}

async function fetchCreationDate(folderName) {
    try {
        console.log(`📅 Fetching creation date for ${folderName}...`);
        const commitsUrl = `https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/commits?path=${folderName}&per_page=100&ref=${REPO_BRANCH}`;
        const commits = await fetchWithAuth(commitsUrl);
        
        if (commits && commits.length > 0) {
            // Get the earliest commit for this folder
            const earliestCommit = commits[commits.length - 1];
            const creationDate = earliestCommit.commit.author.date;
            console.log(`📅 Found creation date for ${folderName}: ${creationDate}`);
            return creationDate.split('T')[0]; // Return just the date part
        }
    } catch (error) {
        console.warn(`⚠️ Could not fetch creation date for ${folderName}:`, error.message);
    }
    
    // Fallback to current date
    return new Date().toISOString().split('T')[0];
}

async function generateSmolHubData() {
    try {
        // Ensure _data directory exists
        const dataDir = path.dirname(OUTPUT_FILE);
        if (!fs.existsSync(dataDir)) {
            fs.mkdirSync(dataDir, { recursive: true });
        }
        
        // Fetch repository contents
        const directories = await fetchRepositoryContents();
        
        // Process each directory
        console.log('📝 Processing playground directories...');
        const projects = [];
        
        for (let i = 0; i < directories.length; i++) {
            const folder = directories[i];
            const progress = `(${i + 1}/${directories.length})`;
            console.log(`🎮 Processing ${folder.name} ${progress}...`);
            
            const { description, readmeContent } = await fetchReadmeContent(folder.name);
            const creationDate = await fetchCreationDate(folder.name);
            const projectTags = generateProjectTags(folder.name, description, readmeContent);
            
            projects.push({
                name: folder.name,
                display_name: formatDisplayName(folder.name),
                description: description,
                readme_content: readmeContent,
                tags: projectTags,
                github_url: `https://github.com/${REPO_OWNER}/${REPO_NAME}/tree/${REPO_BRANCH}/${folder.name}`,
                api_url: folder.url,
                download_url: folder.download_url,
                created_date: creationDate,
                github_date: creationDate
            });
            
            // Small delay to be respectful to GitHub API
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        // Create final data structure
        const smolhubData = {
            last_updated: new Date().toISOString(),
            total_projects: projects.length,
            projects: projects.sort((a, b) => a.display_name.localeCompare(b.display_name))
        };
        
        // Write to file
        fs.writeFileSync(OUTPUT_FILE, JSON.stringify(smolhubData, null, 2), 'utf8');
        
        console.log('✅ SmolHub playground data generated successfully!');
        console.log(`🎮 Total projects: ${projects.length}`);
        console.log(`💾 File saved: ${OUTPUT_FILE}`);
        console.log(`🕒 Last updated: ${smolhubData.last_updated}`);
        
        return smolhubData;
        
    } catch (error) {
        console.error('❌ Error generating SmolHub data:', error);
        process.exit(1);
    }
}

// Run the generator
generateSmolHubData().then(() => {
    console.log('🎉 Done! You can now commit and push the updated _data/smolhub_playground.json file.');
});
