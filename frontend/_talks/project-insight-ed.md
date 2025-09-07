---
title: "Insight-Ed (HackNITR 5.0) | EdTech Platform for Student and Teacher"
collection: talks
excerpt: "Imagine an online classroom where teachers instantly know when and why students lose focus. Our AI-powered solution bridges the knowledge gap by detecting student emotions and attentiveness, highlighting problem areas, makes the teacher aware of each student's progress. With features like reverse video search, dynamic questionnaires, and advanced Q&A bots, we transform the learning experience, making it more interactive and insightful. Redefining online education with our model, ensuring every student gets the attention they need, right when they need it."
type: "Hackathon"
permalink: 
venue: "NIT"
date: March 3 '24
location: "Rourkela"
date_iso: 2024-03-03
---

<div class="project-links" style="display:flex; gap:10px; flex-wrap:wrap; margin: 8px 0 24px;">
  <a href="https://github.com/YuvrajSingh-mist/Insight-Ed" target="_blank" rel="noopener" class="model-details-btn" style="background: linear-gradient(135deg, #111827, #374151); padding: 8px 12px; border-radius: 8px; color: #fff; text-decoration: none;">
    🐙 GitHub
  </a>
</div>

# Insight-Ed


Our product (**website**) provides a solution to tackle the challenge of distance learning between student and teacher, especially in the pandemic era where offline learning was a thing of the past.

Our solution tries to bridge the knowledge gap between a teacher and a student in online classes by leveraging the power of AI to provide the teacher and student with insights to better prepare themselves for the course/lecture, leading to our solution - *Insight-Ed*.


![logo-32](https://github.com/YuvrajSingh-mist/Insight-Ed/assets/141050962/c32ab628-141b-4439-814e-65002f504a5f)

#### Overview of all [ENDPOINTS]
[ENDPOINT.pdf](https://github.com/YuvrajSingh-mist/Insight-Ed/files/15175072/ENDPOINT.pdf)



## Tech Stack

**Client:** Tailwind CSS, TypeScript, Next.js

**Backend:** Flask, Python, Langchain,  GeminiPro and GeminiProVision API, OpenAI Whisper API, HuggingFace Open-Source Models (EmotionLLM), MTCNN, Next.js, TypeScript

**Storage:** PostgreSQL, ChromaDB

**Other Tools:** GCP (Cloud Run), Docker, Vercel

## Features

1. **Detection of Emotion and Attentiveness of a Student**

Used Transfer Learning on EfficientNetB7 model along with a custom dataset combined with an open-sourced LLMs to capture the same through the recorded lecture which the student attended through the webcam during that session.

2. **Topic Modelling for Each Segment where Student lost attention**

-> Through the power of Gemini-Pro SDK and OpenAI Whisper module, each of the segments where the loss of interest is detected is transcribed into text and the most relevant keywords(Topic Modelling) are done to bring out the topics where students need clarification.
-> The teacher is thus present with an analysis as to when the student lost his/her attention and on which topic(s) through the video lecture

3. **Generation of Different Topics on specified time intervals- Saving Teacher's Time**

-> This feature allows automatic Generation of Different Topics-Topic Modelling depending on a hyperparameter(time interval) chosen by the teacher exclusively.

4. **Reverse Video Search**

-> This feature allows the student to not get through the video again and again for a particular keyword but rather type it and get the exact timestamp(s) for the same.

5. **Generation of Questionnaire on topics where students lacked attention**

-> This feature allows the teacher to generate/upload a questionnaire through an LLM for the topics for the student to double-check his/her understanding to plan further steps.
-> The student can also get recommended most relevant YouTube videos on the same or the teacher could suggest some materials or videos from their course itself.

6. **Advance RAG-based QnA bot (Handwritten/Non-Handwritten)**
-> This feature allows the user to perform question-answer with their uploaded PDFs whether it's Handwritten/Non-Handwritten (under-development).
-> This allows the student to quickly search and get relevant information/explanation for a particular topic of interest and not go through searching for it, wasting their precious time.
   
7. **Advance RAG-based QnA bot for Video/Tutorial**

-> This feature allows the user to perform question-answer with their video/tutorials.
-> This allows the student to quickly search and get relevant information/explanation for a particular topic of interest in the video/tutorial and not go through searching for it throughout the video.
   
## Requirements


```bash
Docker for Desktop
```

## Testing

### Server

Pull the server image from DockerHub

```bash
  docker pull yuvrajsingh9886/insight-ed-server:v1.0
```

### Database Setup

Create an environment variable 'DATABASE_URL' in the .env file for the database to work

```bash
  DATABASE_URL="postgresql://swataswayamdash:Dcnhf1IajsW5@ep-gentle-math-a5z04t41.us-east-2.aws.neon.tech/Insight-Ed?sslmode=require"
```
### Gemini Setup

Create an environment variable 'GOOGLE_API_KEY' in the .env file for GeminiPro to work

```bash
  GOOGLE_API_KEY = "YOUR_SECRET KEY"
```


Run the server image

```bash
  docker container run -d -p 8080:8080  yuvrajsingh9886/insight-ed-server:v1.0
```

### Website

Clone the project

```bash
  git clone https://github.com/swataswayam-14/Insight-Ed
```

Go to the project directory

```bash
  cd Insight-Ed
```



## License

[MIT](https://choosealicense.com/licenses/mit/)


## Authors


- [@Swata Swayam Dash](https://github.com/swataswayam-14)
- [@YuvrajSingh](https://github.com/YuvrajSingh-mist)


