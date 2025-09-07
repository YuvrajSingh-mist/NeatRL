#!/usr/bin/env node

/**
 * Secure Models Data Generator
 * This script fetches models data from GitHub using your personal access token
 * and generates the _data/models.json file for Jekyll to use.
 * 
 * Usage: GITHUB_TOKEN=your_token_here node generate-models-data.js
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

// Check for GitHub token
const GITHUB_TOKEN = process.env.GITHUB_TOKEN;
if (!GITHUB_TOKEN) {
    console.error('❌ Error: GITHUB_TOKEN environment variable is required');
    console.log('💡 Usage: GITHUB_TOKEN=your_token_here node generate-models-data.js');
    process.exit(1);
}

// Configuration
const REPO_OWNER = 'YuvrajSingh-mist';
const REPO_NAME = 'Paper-Replications';
const OUTPUT_FILE = path.join(__dirname, '_data', 'models.json');

console.log('🚀 Starting models data generation...');
console.log(`📂 Repository: ${REPO_OWNER}/${REPO_NAME}`);
console.log(`📄 Output file: ${OUTPUT_FILE}`);

async function fetchWithAuth(url) {
    const response = await fetch(url, {
        headers: {
            'Authorization': `token ${GITHUB_TOKEN}`,
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'Portfolio-Models-Generator'
        }
    });
    
    if (!response.ok) {
        throw new Error(`GitHub API error: ${response.status} ${response.statusText}`);
    }
    
    return response.json();
}

async function fetchRepositoryContents() {
    console.log('📡 Fetching repository contents...');
    const url = `https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/contents/`;
    const contents = await fetchWithAuth(url);
    
    // Filter only directories
    const directories = contents.filter(item => item.type === 'dir');
    console.log(`📁 Found ${directories.length} model directories`);
    
    return directories;
}

async function fetchReadmeContent(folderName) {
    try {
        const url = `https://api.github.com/repos/${REPO_OWNER}/${REPO_NAME}/contents/${encodeURIComponent(folderName)}/README.md`;
        const readmeData = await fetchWithAuth(url);
        const readmeContent = Buffer.from(readmeData.content, 'base64').toString('utf8');
        
        // Extract description from README (first non-header line)
        const lines = readmeContent.split('\n').filter(line => line.trim() !== '');
        let description = '';
        for (let line of lines) {
            if (!line.startsWith('#') && line.trim() !== '') {
                description = line.trim();
                break;
            }
        }
        
        return {
            description: description || `Advanced AI model implementation for ${folderName.toLowerCase()}.`,
            readmeContent: readmeContent
        };
    } catch (error) {
        console.log(`⚠️  No README found for ${folderName}, using default description`);
        return {
            description: `Advanced AI model implementation for ${folderName.toLowerCase()}.`,
            readmeContent: `# ${folderName}\n\nThis model is part of the Paper-Replications project.\n\n[View on GitHub](https://github.com/${REPO_OWNER}/${REPO_NAME}/tree/master/${folderName})`
        };
    }
}

function formatDisplayName(folderName) {
    return folderName
        .replace(/[-_]/g, ' ')
        .split(' ')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
        .join(' ');
}

async function generateModelsData() {
    try {
        // Ensure _data directory exists
        const dataDir = path.dirname(OUTPUT_FILE);
        if (!fs.existsSync(dataDir)) {
            fs.mkdirSync(dataDir, { recursive: true });
        }
        
        // Fetch repository contents
        const directories = await fetchRepositoryContents();
        
        // Process each directory
        console.log('📝 Processing model directories...');
        const models = [];
        
        for (let i = 0; i < directories.length; i++) {
            const folder = directories[i];
            const progress = `(${i + 1}/${directories.length})`;
            console.log(`📦 Processing ${folder.name} ${progress}...`);
            
            const { description, readmeContent } = await fetchReadmeContent(folder.name);
            
            models.push({
                name: folder.name,
                display_name: formatDisplayName(folder.name),
                description: description,
                readme_content: readmeContent,
                github_url: `https://github.com/${REPO_OWNER}/${REPO_NAME}/tree/master/${folder.name}`,
                api_url: folder.url,
                download_url: folder.download_url
            });
            
            // Small delay to be respectful to GitHub API
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        // Create final data structure
        const modelsData = {
            last_updated: new Date().toISOString(),
            total_models: models.length,
            models: models.sort((a, b) => a.display_name.localeCompare(b.display_name))
        };
        
        // Write to file
        fs.writeFileSync(OUTPUT_FILE, JSON.stringify(modelsData, null, 2), 'utf8');
        
        console.log('✅ Models data generated successfully!');
        console.log(`📊 Total models: ${models.length}`);
        console.log(`💾 File saved: ${OUTPUT_FILE}`);
        console.log(`🕒 Last updated: ${modelsData.last_updated}`);
        
        return modelsData;
        
    } catch (error) {
        console.error('❌ Error generating models data:', error);
        process.exit(1);
    }
}

// Run the generator
generateModelsData().then(() => {
    console.log('🎉 Done! You can now commit and push the updated _data/models.json file.');
});
