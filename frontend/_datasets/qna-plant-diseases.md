---
title: QnA Plant Diseases Dataset
excerpt: "This dataset contains a comprehensive Question & Answer collection focused on plant diseases, their management, treatment protocols, and diagnostic techniques. The dataset provides detailed information about various plant pathologies, fungicide applications, and disease identification methods for ag<br/><img src='/images/500x300.png'>"
collection: datasets
github_url: https://github.com/YuvrajSingh-mist/Datasets-Collection/tree/main/QnA-Plant-Diseases
size: 4,283 Q&A pairs
format: CSV
samples: 4,283
license: Open Source
last_updated: 2025-08-08T13:46:27.618071
tags:
  - Qna
  - Plant
  - Diseases
tasks:
  - Computer Vision
  - Detection
  - Machine Learning
  - Ner
  - Nlp
---

---

## Overview

This dataset contains a comprehensive Question & Answer collection focused on plant diseases, their management, treatment protocols, and diagnostic techniques. The dataset provides detailed information about various plant pathologies, fungicide applications, and disease identification methods for agricultural and horticultural applications.

## Dataset Information

- **File Name**: `qna-plant-diseases.csv`
- **Total Records**: 4,283 Q&A pairs
- **Format**: CSV (Comma Separated Values)
- **Encoding**: UTF-8

## Data Structure

The dataset contains the following columns:

| Column | Description |
|--------|-------------|
| `Index` | Sequential record identifier (0-based indexing) |
| `ANSWER` | The answer to the corresponding question |
| `QUESTION.question` | The actual question being asked |
| `QUESTION.paragraph` | The source paragraph or context from which the Q&A pair was extracted |

## Content Categories

The dataset covers comprehensive aspects of plant disease management:

### 1. **Disease Treatment Protocols**
- Fungicide applications and dosages
- Treatment schedules and frequency
- Root feed methodologies
- Chemical concentration specifications

### 2. **Disease Identification**
- Visual diagnostic techniques
- Symptom recognition and description
- Healthy vs. infected plant comparisons
- Fruiting body identification

### 3. **Chemical Treatments**
- Tridomorph 75% EC applications
- Dosage calculations (2ml in 100ml water)
- Application methods and timing
- Treatment duration (typically 3-month cycles)

### 4. **Plant Health Assessment**
- Visual indicators of plant health
- Disease progression monitoring
- Comparative analysis techniques
- Early detection methods

## Key Topics Covered

### **Fungicide Management**
- **Tridomorph Applications**: Detailed protocols for 75% EC concentration
- **Root Feed Systems**: Monthly application schedules over 3-month periods
- **Dosage Precision**: Exact measurements (2ml per 100ml water)
- **Treatment Timing**: Monthly intervals for optimal effectiveness

### **Disease Diagnosis**
- **Visual Inspection**: Stem base examination techniques
- **Symptom Recognition**: Identification of fruiting bodies and infection signs
- **Comparative Analysis**: Healthy vs. infected plant characteristics
- **Early Detection**: Preventive monitoring strategies

### **Plant Health Management**
- **Preventive Measures**: Proactive treatment approaches
- **Treatment Regimens**: Structured 3-month treatment cycles
- **Health Monitoring**: Regular assessment protocols
- **Maintenance Practices**: Ongoing plant care strategies

## Data Quality

- ✅ **Structured Format**: Consistent CSV structure with comprehensive coverage
- ✅ **Technical Accuracy**: Precise dosage and application information
- ✅ **Visual Context**: References to diagnostic images and visual indicators
- ✅ **Comprehensive Coverage**: 4,283+ Q&A pairs covering diverse plant disease topics
- ✅ **Practical Application**: Real-world treatment protocols and procedures

## Use Cases

This dataset is ideal for:

### 1. **Agricultural AI and Machine Learning**
- Plant disease diagnostic systems
- Agricultural chatbots and virtual assistants
- Automated treatment recommendation engines
- Computer vision training for disease detection

### 2. **Educational Applications**
- Plant pathology training materials
- Agricultural extension education
- Veterinary and agricultural curriculum development
- Professional development programs

### 3. **Research and Development**
- Disease management effectiveness studies
- Treatment protocol optimization research
- Comparative analysis of fungicide applications
- Agricultural technology development

### 4. **Digital Agriculture Solutions**
- Mobile applications for farmers
- Decision support systems
- Knowledge management platforms
- Expert system development

## Sample Data

<div class="table-responsive">
<table class="table table-striped">
<thead>
<tr>
<th>ANSWER</th>
<th>QUESTION.question</th>
<th>QUESTION.paragraph</th>
</tr>
</thead>
<tbody>
<tr>
<td>Root feed with tridomorph 75% EC 2ml in 100ml water once in a month for 3 months.</td>
<td>What is the root feed recommendation for managing tree diseases?</td>
<td>Root feed with tridomorph 75% EC 2ml in 100ml water once in a month for 3 months</td>
</tr>
<tr>
<td>75% EC.</td>
<td>What is the concentration of tridomorph used in the root feed?</td>
<td>Root feed with tridomorph 75% EC 2ml in 100ml water once in a month for 3 months</td>
</tr>
<tr>
<td>The fruiting bodies are located at the base of the infected tree.</td>
<td>Where are the fruiting bodies of the infected tree located?</td>
<td>Fruiting bodies at base of the infected tree</td>
</tr>
</tbody>
</table>
</div>

---

*This dataset is intended for educational, research, and development purposes. Always consult with qualified agricultural professionals before implementing any treatment protocols in real-world applications.*
