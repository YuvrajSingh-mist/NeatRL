---
title: "FarmGenie (GeoHack 2024) | Empowering farmers with real-time insights and expert guidance via AI-driven space"
collection: talks
excerpt: "
Our platform utilizes LLMs and a Mixture of Expert (MoE) approaches to provide precise guidance on
soil management, plant disease identification, and irrigation techniques. Built as a scalable web application with a
Next.js frontend and backend, and supported by a Redis queue and multiple worker nodes, FarmGenie ensures
robust performance. The system’s multilingual support, interactive community forum, and up-to-date knowledge
base facilitate seamless, expert-driven assistance for both new and experienced farmers.
"
type: "Hackathon"
permalink: 
venue: "IIIT-BH"
date: July 19 '24
location: Bhubaneswar
date_iso: 2024-07-19
---

<div class="project-links" style="display:flex; gap:10px; flex-wrap:wrap; margin: 8px 0 24px;">
  <a href="https://github.com/YuvrajSingh-mist/FarmGenie/tree/main" target="_blank" rel="noopener" class="model-details-btn" style="background: linear-gradient(135deg, #111827, #374151); padding: 8px 12px; border-radius: 8px; color: #fff; text-decoration: none;">
    🐙 GitHub
  </a>
</div>

# FarmGenie 

--- 
- **Our project only supports conversation (input/output) in Hindi/English for now**.
- FarmGenie Official Website: http://64.227.135.219:3000/  (Production removed: test locally using docker or setting up locally)
- FarmGenie Community Forum: https://farmgenie-rho.vercel.app/
- FarmGenie Agri-Schemes Website: https://farmgenie-agrischemes.vercel.app/
- Please refer to 'MoE-Framework.png' for our proposed framework
---

### Vision

In the ever-evolving agricultural landscape, farmers often face challenges in accessing up-to-date knowledge and resources to improve their farming practices. To address this, we have developed a comprehensive product that leverages the power of large language models (LLMs) and advanced conversational AI to create an interactive platform for farmers.


### Goal

The primary goal of this project is to design a product that helps farmers, especially those new to the field, gain up-to-date knowledge on various agricultural and farming practices from the Internet through a user-friendly interface with an AI-based conversation bot.

---

### Seamless Interface

##### FarmGenie Official Website 

- http://64.227.135.219:3000/

- The product is built as a web application using a Next.js client, and a Next.js backend, leveraging a Redis queue and multiple worker nodes to ensure scalability and robustness. The knowledge base is curated from openly available PDF books and reports covering various agricultural topics.

- Dual Marketplace:
  - Input Marketplace (for local sellers and dealers)
  - Output Marketplace (for farmers)


##### FarmGenie Conversational-AI

###### Framework

- The product's core is a sophisticated conversational AI framework combining a Mixture of Experts (MoE) approach, Retrieval-Augmented Generation (RAG), and language translation capabilities. 

- The MoE component consists of three fine-tuned small language models (SLMs) that have become specialized experts in soil management, plant disease identification, and irrigation techniques, allowing them to handle complex queries by breaking them down into subproblems.

- The "conversation managing" LLMs classify the user's queries, manage the flow of the conversation, and facilitate the translation between the user's preferred language and English, ensuring seamless interaction for farmers. 


##### FarmGenie Community-Forum

- https://farmgenie-rho.vercel.app/

- Additionally, the product aims to create an interactive platform where farmers can connect, share their knowledge, and learn from their peers, fostering a community-driven approach to agricultural knowledge dissemination.

###### Features
  - Multilingual Support.
  - Gamified help system.
  - Communities.


##### FarmGenie Marketplace-Chatbot

###### Framework

- This is developed to redefine the way customers interact with online retail shops by replacing the traditional "search and find" method with Natural Language-based instructions which gives a sense of control and connectivity with ease of information retrieval for products, just like in a real shop with LLM Agents as shopkeepers.
  
- The product's core is a sophisticated conversational AI framework that leverages the power of LLM as Agents with the combination of Gemini and OpenAI GPT-4o.

- The main motivation is to motivate and allow local retail shops to increase their sales through the formation of a shared local network of shop owners who can increase their reach to local people. This is also advantageous for small-scale farmers who don't have many means to get cheap and nearby products by making them aware that the items they are searching for could very well be nearby.  

- The "conversation managing" LLMs classify the user's queries, manage the flow of the conversation, and facilitate the translation between the user's preferred language and English, ensuring seamless interaction for farmers.

###### Features
  - Many! since it depends on the user's language of query, major areas being Reviews, Nearby Location Access, Product-Search and its respective Product Information on which any queries could be performed.
    
  
##### FarmGenie AgriculturalSchemes Website

- https://farmgenie-agrischemes.vercel.app/

###### Features
  - Empowering Farmers: A dedicated platform designed to provide farmers with essential information on government schemes for agriculture and farming.
  - User-Friendly Interface: Intuitive navigation allows for easy access to a wealth of resources, ensuring a seamless experience for users.
  - Comprehensive Database: Explore a wide variety of initiatives, including subsidies, grants, and training programs tailored to support farmers.
  - Regular Updates: Stay informed with the latest changes and updates to government policies and schemes relevant to agriculture.
  - Helpful Guides: Access informative guides and tips to help farmers maximize the benefits of available programs and resources.


##### Website
 
- The front end is built using Next.js, a React framework that provides server-side rendering, static site generation, and other performance optimizations. The backend is also built using Next.js which handles the communication between the front end and the ML backend. 

The Next.js backend is responsible for the following tasks:

1.      **Request Handling**: The Next.js server receives all the requests from the front end, such as user queries, and passes them to the Redis queue.
2.      **Redis Queue**: A Redis queue is used to manage incoming requests. This helps distribute the load across multiple worker node processes, ensuring the ML backend does not get overwhelmed.
3.      **Worker node Processes**: Multiple worker nodes are set up to consume the requests from the Redis queue. These processes then pass the requests to the ML backend for processing.
4.      **ML Backend Integration**: The Next.js backend communicates with the ML backend, which is responsible for handling the various ML tasks, such as query classification, subproblem generation, and retrieval-augmented generation (RAG).
5.      **Database Integration**: The website uses a single PostgreSQL database, provided by Neon DB, as the single source of truth. This database stores all the user, farmer, and retailer data, as well as the content for the official Farmgenie website.
6.      **User Management**: The Next.js backend handles user management, including authentication, authorization, and user data storage in the PostgreSQL database (neon tech).
7.      **Scalability**: The use of a Redis queue and multiple worker node processes ensures the application can scale to handle a large number of concurrent user requests without overloading the ML backend.
8.      **Deployment**: The entire application, including the frontend, backend, and ML components, is packaged and deployed using Docker containers, ensuring consistent and reliable deployment across different environments.
9.      **Monitoring and Logging**: The application includes comprehensive monitoring and logging solutions to track performance, errors, and user activity, enabling the team to identify and resolve issues quickly.
10.     **Security**: The website incorporates industry-standard security practices, such as SSL/TLS encryption, user authentication, and input validation, to protect user data and prevent unauthorized access.



## Tech Stack

**Client**: Tailwind CSS, TypeScript, Next.js

**Backend**: Flask, Python, Langchain, GeminiPro API, peft , bitsandbytes, transformers

**Storage/VectoStore**: PostgreSQL, FAISS, Pinecone, MySQL

**Other Tools**: Unsloth, GCP, Docker, Vercel, DigitalOcean


## Fine Tuning

**Family**: Phi (Phi3-128k-instruct) \
**Fine-tuned Models**: [Models Link (HuggingFace)](https://huggingface.co/YuvrajSingh9886) \
**Dataset**: [Datasets Link (HuggingFace)](https://huggingface.co/YuvrajSingh9886?sort_datasets=likes#datasets) \
**Steps**: 1000 \
**Batch size**: 8 \
**Alpha**: 2e-5/2e-3 \
**Hardware**: NVIDIA RTX 4050 6GB VRAM (GPU), Inten CORE i7 16 GB RAM (CPU) 



## Requirements

```bash
docker (Docker Desktop preferred since it comes with docker-compose inbuilt)

GPU with NVIDIA CUDA Drivers==12.2

GPU with NVIDIA CUDA Drivers==12.2

Make sure NVIDIA Container Toolkit-
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html
```

### Run only Chatbots (FarmGenie (main) and Marketplace - Streamlit version)

```bash
git clone https://github.com/YuvrajSingh-mist/FarmGenie.git
cd FarmGenie/farmgenie_hf/api
```

Create a .env file and make sure to input the following api keys in the current directory

```bash
GEMINI_API_KEY=<YOUR_API_KEY_HERE> \
OPENAI_API_KEY=<YOUR_API_KEY_HERE> \
hf_token=<YOUR_API_KEY_HERE> \
COHERE_API_KEY=<YOUR_API_KEY_HERE> \
MAPBOX_API_KEY=<YOUR_API_KEY_HERE> \
PINECONE_API_KEY=<YOUR_API_KEY_HERE> \
GOOGLE_MAPS_API=<YOUR_API_KEY_HERE> \


```
Open a terminal within the same directory and execute the following command
```bash
cd ..
docker-compose up --build
```

```bash
You'll see it running on localhost:8000/docs
```


## Run Locally

Using Docker

To run the FarmGenie official website Docker image locally, start by pulling the image from Docker Hub using the command:

```bash
  docker pull swayam14/farmwebappsecret:v6.0
```


This command downloads the specified image to your local machine, allowing you to run it without needing to build it from scratch. Once the image is successfully pulled, you can start the container using the following command, which maps port 3001 on your local machine to port 3001 in the container:

```bash
  docker run -p 3001:3001 swayam14/farmwebappsecret:v6.0 
```
This setup enables you to access the FarmGenie application locally via http://localhost:3001, allowing for easy testing and development of the application as if it were running on a remote server.


Clone the project

```bash
  git clone https://github.com/swataswayam-14/FarmGenie
```

Go to the project directory

```bash
  cd FarmGenie
```

Go to the next-app directory

```bash
  cd next-app
```

Install dependencies

```bash
  npm install
```

Generate prisma client

```bash
  npx prisma generate
```

Start the server

```bash
  npm run start
```

Go to the communtiy-forum directory

```bash
  cd community-forum
```

Install dependencies

```bash
  npm install
```

Start the server

```bash
  npm run start
```


## 🔗 Links

- [HuggingFace Account](https://huggingface.co/YuvrajSingh9886)
- [Youtube Video - FarmGenie Overview](https://youtu.be/KXKhoLb8EfA)


## Authors

- [@Swata Swayam Dash](https://github.com/swataswayam-14)
- [@YuvrajSingh](https://www.github.com/YuvrajSingh-mist)
