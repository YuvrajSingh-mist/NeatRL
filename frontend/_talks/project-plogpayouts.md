---
title: "PlogPayouts | AI-driven Plogging System"
collection: talks
type: "Google's Solution Challenge '24"
excerpt: "Transform your daily jog into a mission for a cleaner world with PlogPayouts. Our innovative website + app rewards you for collecting litter, promoting fitness and environmental cleanliness. Utilizing AI for trash categorization and optimized routes, and fostering community through shared stories, PlogPayouts turns every step into a step towards a greener, more inclusive society. Join us and make a difference today! "
permalink: 
venue: "IIIT"
date: December '23 - March '24
location: "Bhubaneswar"
date_iso: 2023-12-01
---

<div class="project-links" style="display:flex; gap:10px; flex-wrap:wrap; margin: 8px 0 24px;">
  <a href="https://github.com/YuvrajSingh-mist/PlogPayouts" target="_blank" rel="noopener" class="model-details-btn" style="background: linear-gradient(135deg, #111827, #374151); padding: 8px 12px; border-radius: 8px; color: #fff; text-decoration: none;">
    🐙 GitHub
  </a>
</div>

# PlogPayouts  

*GDSC Soltuion Challenge 2024*

To bring about an effective change in the garbage collection process and also tackle the social stigma regarding the position and work of garbage collectors to harbour a cleaner and greener environment, working together without any social differences.



<img src="https://firebasestorage.googleapis.com/v0/b/fashionx-ebe6c.appspot.com/o/temp%2Flogo_1.png?alt=media&token=13098b57-3f6a-4dc3-9d1e-2bd35613b814" width="400" height="300">  <img src="https://firebasestorage.googleapis.com/v0/b/fashionx-ebe6c.appspot.com/o/temp%2Flogo_2.png?alt=media&token=a45c4618-9b64-45e3-8491-896d06329af1" width="270" height="270">





## Demo

#### Website (Preview of Features)

[Website](https://firebasestorage.googleapis.com/v0/b/fashionx-ebe6c.appspot.com/o/temps%2FDocument%20-%20Google%20Chrome%202024-02-13%2001-28-28.mp4?alt=media&token=f9f162b8-9bd9-41a1-ad81-fcfbdd687942)


#### App (Preview of Features)

[App](https://www.youtube.com/watch?v=Dn2QWmRpGrs)


#### YouTube (Project Demo)

[Youtube Video Link](https://youtu.be/fU7S5YPEcQk)



## Tech Stack

**Client:** Flutter, HTML, CSS JavaScript, Bootstrap

**Backend:** Flask, Python, Django, Keras, OpenCV, YOLOv8

**Storage:** Firebase Firestore, Realtime Database, SQLite

**Other Tools:** GCP VM Instances, Mapbox, Render, Docker

## Features

1. **Categorization of Trash w/ Deep Learning**

Used Transfer Learning on the VGG19 model along with a custom dataset combined with open-sourced and hand-labelled ones.

2. **Reward System**

-> Earn points for each **category** and **count** (using YOLOv8) of litter collected during plogging and once the garbage collected has been verified by the Garbage Collector on-site (so the garbage collected does reach to a factory through the worker), points are sent.

-> Redeem points in the 'Store' for gift cards.

-> The categorization of garbage is intended for smoothing out the process of post-garbage collected by already organizing the garbage as much as possible.


3. **Map Location**

Genetic Algorithm automatically finds the best route for the locations of trash sites marked as 'Favourite' so as to cater to their jogging times (up to 3 max).

4. **Blog System**

Share plogging experiences, environmental tips, and success stories through the integrated blog system, bringing the community together and breaking the social stigma of how 'garbage collectors' are perceived.



## Requirements


Please install the following requirements before proceeding with the next steps
#### Server

```bash
Docker
```
    



#### App


```bash
Android Studio
Flutter SDK
```
    
    
1. Ensure that you are able to run Flutter apps locally by  installing the following packages
2. Run the following command to ensure your system meets the requirements to run the app. All requirements must be met to run the app.

        flutter doctor 



## Run Locally

### Server

Pull the server image from DockerHub

```bash
  docker pull yuvrajsingh9886/plogpayouts-server:v1.0
```

Run the server image

```bash
  docker container run -d -p 8080:8080 yuvrajsingh9886/plogpayouts-server:v1.0
```

### Website

Clone the project

```bash
  git clone https://github.com/anurag6569201/PlogPayouts.git
```

Go to the project directory

```bash
  cd webapp
```

Create a virtual environment

```bash
  mkvirtualenv envname
```

Activate virtualenv

```bash
  workon envname
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Run the website Locally

```bash
  python manage.py runserver
```






### App

1. Run the following commands to download all required dependancies.

```bash
flutter clean
flutter pub get
```

2. Run the following command to run the app on an emulator/real device

```bash
flutter run
```

## License

[MIT](https://choosealicense.com/licenses/mit/)


## Authors


- [@AnuragSingh](https://github.com/anurag6569201)
- [@YuvrajSingh](https://github.com/YuvrajSingh-mist)


