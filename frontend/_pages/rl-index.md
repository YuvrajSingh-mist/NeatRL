---
title: "RL From Scratch"
layout: collection
permalink: /rl/
collection: rl
entries_layout: grid
classes: wide
author_profile: false
sidebar:
  nav: "main"
header:
  overlay_color: "#000"
  overlay_filter: "0.5"
  overlay_image: /assets/images/rl-hero.jpg
  actions:
    - label: "View on GitHub"
      url: "https://github.com/YuvrajSingh-mist/Reinforcement-Learning"
excerpt: "A comprehensive collection of reinforcement learning algorithms implemented from scratch"
intro:
  - excerpt: "Explore our collection of reinforcement learning algorithms, each implemented from scratch with detailed explanations, visualizations, and performance metrics."
feature_row:
  - image_path: /assets/images/dqn-thumb.png
    alt: "Deep Q-Networks"
    title: "Deep Q-Networks (DQN)"
    excerpt: "Value-based reinforcement learning using deep neural networks"
    url: "/rl/dqn-dqn/"
    btn_label: "Learn More"
    btn_class: "btn--primary"
  - image_path: /assets/images/ppo-thumb.png
    alt: "Proximal Policy Optimization"
    title: "Proximal Policy Optimization (PPO)"
    excerpt: "State-of-the-art policy gradient method with clipped surrogate objective"
    url: "/rl/ppo-ppo/"
    btn_label: "Learn More"
    btn_class: "btn--primary"
  - image_path: /assets/images/a2c-thumb.png
    alt: "Advantage Actor-Critic"
    title: "Advantage Actor-Critic (A2C)"
    excerpt: "Synchronous advantage actor-critic method combining policy and value learning"
    url: "/rl/a2c-a2c/"
    btn_label: "Learn More"
    btn_class: "btn--primary"
---

{% include feature_row id="intro" type="center" %}

<div class="rl-gallery">
  <div class="rl-filters">
    <button class="filter-btn active" data-filter="all">All Algorithms</button>
    <button class="filter-btn" data-filter="value-based">Value-Based</button>
    <button class="filter-btn" data-filter="policy-based">Policy-Based</button>
    <button class="filter-btn" data-filter="actor-critic">Actor-Critic</button>
    <button class="filter-btn" data-filter="multi-agent">Multi-Agent</button>
  </div>

  <div class="rl-grid">
    {% for post in site.rl %}
      {% assign categories = post.categories | join: ' ' | downcase %}
      <div class="rl-card" data-category="{{ categories }}">
        <div class="rl-card-header">
          {% if post.environment %}
            <div class="rl-card-env">{{ post.environment }}</div>
          {% endif %}
          <h3 class="rl-card-title">
            <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
          </h3>
        </div>
        
        <div class="rl-card-content">
          <p class="rl-card-excerpt">{{ post.excerpt | markdownify | strip_html | truncate: 120 }}</p>
          
          <div class="rl-card-meta">
            {% if post.framework %}
              <span class="rl-tag framework">{{ post.framework }}</span>
            {% endif %}
            {% if post.category %}
              <span class="rl-tag category">{{ post.category }}</span>
            {% endif %}
          </div>
          
          {% if post.categories %}
            <div class="rl-card-tags">
              {% for category in post.categories %}
                <span class="rl-tag">{{ category }}</span>
              {% endfor %}
            </div>
          {% endif %}
        </div>
        
        <div class="rl-card-footer">
          <a href="{{ post.url | relative_url }}" class="rl-card-link">
            View Implementation
            <i class="fas fa-arrow-right"></i>
          </a>
          {% if post.github_url %}
            <a href="{{ post.github_url }}" class="rl-card-github" target="_blank">
              <i class="fab fa-github"></i>
            </a>
          {% endif %}
        </div>
      </div>
    {% endfor %}
  </div>
</div>

<style>
.rl-gallery {
  max-width: 1200px;
  margin: 2rem auto;
  padding: 0 1rem;
}

.rl-filters {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-bottom: 2rem;
  justify-content: center;
}

.filter-btn {
  padding: 0.5rem 1rem;
  border: 2px solid #e9ecef;
  background: white;
  border-radius: 25px;
  cursor: pointer;
  transition: all 0.3s ease;
  font-weight: 500;
  font-size: 0.9rem;
}

.filter-btn:hover,
.filter-btn.active {
  background: #007bff;
  color: white;
  border-color: #007bff;
  transform: translateY(-2px);
}

.rl-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
  gap: 2rem;
  margin-top: 2rem;
}

.rl-card {
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
  overflow: hidden;
  border: 1px solid #e9ecef;
  display: flex;
  flex-direction: column;
}

.rl-card:hover {
  transform: translateY(-8px);
  box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
}

.rl-card-header {
  padding: 1.5rem 1.5rem 1rem;
  position: relative;
}

.rl-card-env {
  position: absolute;
  top: 1rem;
  right: 1rem;
  background: #28a745;
  color: white;
  padding: 0.25rem 0.75rem;
  border-radius: 15px;
  font-size: 0.75rem;
  font-weight: 600;
}

.rl-card-title {
  margin: 0;
  font-size: 1.3rem;
  font-weight: 700;
  line-height: 1.3;
  margin-right: 4rem;
}

.rl-card-title a {
  color: #2c3e50;
  text-decoration: none;
  transition: color 0.3s ease;
}

.rl-card-title a:hover {
  color: #007bff;
}

.rl-card-content {
  padding: 0 1.5rem 1rem;
  flex-grow: 1;
}

.rl-card-excerpt {
  color: #6c757d;
  line-height: 1.6;
  margin-bottom: 1rem;
}

.rl-card-meta {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

.rl-card-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-bottom: 1rem;
}

.rl-tag {
  background: #f8f9fa;
  color: #495057;
  padding: 0.25rem 0.75rem;
  border-radius: 15px;
  font-size: 0.8rem;
  font-weight: 500;
  border: 1px solid #e9ecef;
}

.rl-tag.framework {
  background: #e3f2fd;
  color: #1976d2;
  border-color: #bbdefb;
}

.rl-tag.category {
  background: #f3e5f5;
  color: #7b1fa2;
  border-color: #ce93d8;
}

.rl-card-footer {
  padding: 1rem 1.5rem;
  border-top: 1px solid #e9ecef;
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: #f8f9fa;
}

.rl-card-link {
  color: #007bff;
  text-decoration: none;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  transition: all 0.3s ease;
}

.rl-card-link:hover {
  color: #0056b3;
  transform: translateX(5px);
}

.rl-card-github {
  color: #6c757d;
  font-size: 1.2rem;
  transition: color 0.3s ease;
}

.rl-card-github:hover {
  color: #212529;
}

/* Responsive Design */
@media (max-width: 768px) {
  .rl-grid {
    grid-template-columns: 1fr;
    gap: 1.5rem;
  }
  
  .rl-filters {
    justify-content: flex-start;
  }
  
  .filter-btn {
    font-size: 0.8rem;
    padding: 0.4rem 0.8rem;
  }
  
  .rl-card-title {
    font-size: 1.1rem;
    margin-right: 3rem;
  }
  
  .rl-card-env {
    font-size: 0.7rem;
    padding: 0.2rem 0.6rem;
  }
}

/* Filter Animation */
.rl-card.filtered-out {
  opacity: 0;
  transform: scale(0.8);
  transition: all 0.3s ease;
  pointer-events: none;
}

.rl-card.filtered-in {
  opacity: 1;
  transform: scale(1);
  transition: all 0.3s ease;
  pointer-events: auto;
}
</style>

<script>
document.addEventListener('DOMContentLoaded', function() {
  const filterButtons = document.querySelectorAll('.filter-btn');
  const cards = document.querySelectorAll('.rl-card');

  filterButtons.forEach(button => {
    button.addEventListener('click', function() {
      // Remove active class from all buttons
      filterButtons.forEach(btn => btn.classList.remove('active'));
      // Add active class to clicked button
      this.classList.add('active');

      const filter = this.getAttribute('data-filter');

      cards.forEach(card => {
        const categories = card.getAttribute('data-category');
        
        if (filter === 'all' || categories.includes(filter.replace('-', ' '))) {
          card.classList.remove('filtered-out');
          card.classList.add('filtered-in');
          card.style.display = 'flex';
        } else {
          card.classList.remove('filtered-in');
          card.classList.add('filtered-out');
          setTimeout(() => {
            if (card.classList.contains('filtered-out')) {
              card.style.display = 'none';
            }
          }, 300);
        }
      });
    });
  });

  // Initialize all cards as visible
  cards.forEach(card => {
    card.classList.add('filtered-in');
  });
});
</script>