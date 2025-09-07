---
title: "MoviesMania (Geek-o-thon) | A Reverse Search based Movies Recommendation System"
collection: talks
type: "Hackathon"
permalink: 
excerpt: "Step into the future of entertainment discovery with MoviesMania. The rpoduct aims to simplify your search for the perfect movie or web series. Using various AI/ML techniques and elements, we analyze uploaded video clips to predict movie titles and recommend similar content with an impressive accuracy. Experience flavoured recommendations tailored to your tastes, powered by Keras, Flask, and advanced face recognition algorithms with a full-fledged movie recommendation system."
venue: "IIIT"
date: October 3 '23
location: "Bhubaneswar"
date_iso: 2023-10-03
---

<div class="project-links" style="display:flex; gap:10px; flex-wrap:wrap; margin: 8px 0 24px;">
  <a href="https://github.com/YuvrajSingh-mist/MoviesMania" target="_blank" rel="noopener" class="model-details-btn" style="background: linear-gradient(135deg, #111827, #374151); padding: 8px 12px; border-radius: 8px; color: #fff; text-decoration: none;">
    🐙 GitHub
  </a>
</div>

# Movies Mania

- The project aims to provide an interface for people who want movie recommendations on the basis of some videos uploaded by the user such as a movie clip or YT Short.

- Our project could also provide recommendatios based on any movie Title and on any movie Plot which could be of some other video from our databae of 5000+ movies or fruit of the user's creativity. 


## Authors

- [@Pritam Paul](https://www.github.com/paul-pritam)
- [@Shlok Agrawal](https://www.github.com/agrawal-shlok)
- [@Yuvraj Singh](https://www.github.com/YuvrajSingh-mist)


## Features

- Recommendation based on Videos
- Recommendation based on Title
- Recommendation based on Plot
- Able to download YT Short dirctly on the WebApp through it's link
- Able to get information about a particular movie on our WebApp 

## Tech Stack

**Backend(Logic):** Tensorflow, Keras, Word2Vec, MTCNN, VGGFace, OpenCV, Pickle

**Frontend:** Streamlit (Build)


## Run Locally

```bash
~ Download the required data files from project-files.zip
~ Move the downloaded files from project-files.zip to the MoviesMania folder
~ (REQUIRED) Download Microsoft Edge and its compatible Selenium driver if one wishes to use the Review System
```

Clone the project

```bash
  git clone https://github.com/YuvrajSingh-mist/MoviesMania
```

Go to the project directory

```bash
  cd MoviesMania
```

Install dependencies

```bash
  pip install -r requirements.txt
```


Start the WebApp

```bash
  streamlit run Home.py
```


