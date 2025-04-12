from flask import Blueprint, request, jsonify
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

import wikipedia
import re
import logging
from flask_cors import CORS
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the blueprint
tutor_bp = Blueprint('tutor', __name__)

# Global variables
qa_pipeline = None
summarizer = None
sentence_model = None
topic_knowledge_base = {}

def get_qa_pipeline():
    global qa_pipeline
    if qa_pipeline is None:
        try:
            logger.info("Loading QA model...")
            # Using a more advanced model
            qa_pipeline = pipeline("question-answering", model="deepset/tinyroberta-squad2")
            logger.info("QA model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading QA model: {e}")
            raise e
    return qa_pipeline

def get_summarizer():
    global summarizer
    if summarizer is None:
        try:
            logger.info("Loading summarization model...")
            model_name = "facebook/bart-large-cnn"
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


            tokenizer = AutoTokenizer.from_pretrained(model_name)
            summarizer = {"model": model, "tokenizer": tokenizer}
            logger.info("Summarization model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading summarization model: {e}")
            raise e
    return summarizer

def get_sentence_model():
    global sentence_model
    if sentence_model is None:
        try:
            logger.info("Loading sentence embeddings model...")
            sentence_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')
            logger.info("Sentence embeddings model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading sentence embeddings model: {e}")
            raise e
    return sentence_model

def summarize_text(text, max_length=200):
    try:
        summarizer = get_summarizer()
        model = summarizer["model"]
        tokenizer = summarizer["tokenizer"]
        
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs, max_length=max_length, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        logger.error(f"Error in summarization: {e}")
        # Fallback to simple summarization
        return text[:max_length] + "..."

def get_topic_content(topic):
    global topic_knowledge_base

    if topic in topic_knowledge_base:
        return topic_knowledge_base[topic]

    try:
        search_results = wikipedia.search(topic, results=5)
        if not search_results:
            return f"I couldn't find information about {topic}. Please try a different topic."

        all_content = []
        for result in search_results[:3]:  # Use top 3
            try:
                page = wikipedia.page(result, auto_suggest=False)
                content = page.content
                if content and len(content.split()) > 100:  # Ensure meaningful content
                    all_content.append(content)
                if len(all_content) >= 2:  # Prefer at least 2 sources
                    break
            except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError) as e:
                logger.warning(f"Wikipedia error for {result}: {e}")
                continue

        if not all_content:
            return f"I couldn't find detailed info about {topic}."

        combined_content = "\n\n".join(all_content)
        cleaned_content = re.sub(r'\n==.*?==\n', '\n', combined_content)
        cleaned_content = re.sub(r'\n\n+', '\n\n', cleaned_content)

        full_content = cleaned_content[:20000]  # Extend limit for large topics
        topic_knowledge_base[topic] = full_content
        return full_content

    except Exception as e:
        logger.error(f"Error retrieving topic content: {e}")
        return f"I encountered an error while researching {topic}. Please try again."


sentence_cache = {}

def semantic_search(query, sentences, top_k=5):
    """Find semantically similar sentences using embeddings"""
    try:
        model = get_sentence_model()
        query_embedding = model.encode([query])
        sentence_embeddings = model.encode(sentences)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, sentence_embeddings)[0]
        
        # Get top-k most similar sentences
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [sentences[i] for i in top_indices]
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        # Fallback to simple keyword matching
        keywords = query.lower().split()
        return [s for s in sentences if any(keyword in s.lower() for keyword in keywords)][:top_k]

@tutor_bp.route('/set_topic', methods=['POST'])
def set_topic():
    """Endpoint to set a new topic for tutoring"""
    try:
        data = request.get_json()
        
        # Validate data
        if not data or 'topic' not in data:
            return jsonify({'error': 'No topic provided.'}), 400
        
        topic = data.get('topic')
        
        # Get content for the topic
        content = get_topic_content(topic)
        
        # Generate a better introduction using summarization
        try:
            intro = f"I'll be your tutor for {topic}. Here's a brief overview:\n\n"
            summary = summarize_text(content[:5000], max_length=300)
            intro += summary + "\n\nYou can ask me specific questions about this topic now."
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            # Fallback to simpler intro
            intro = f"I'll be your tutor for {topic}. You can ask me specific questions about this topic now."
        
        return jsonify({
            'success': True,
            'topic': topic,
            'introduction': intro
        })
        
    except Exception as e:
        logger.error(f"Error setting topic: {e}")
        return jsonify({'error': str(e)}), 500

@tutor_bp.route('/ask_question', methods=['POST'])
def ask_question():
    """Improved endpoint to ask questions about the current topic"""
    try:
        data = request.get_json()
        
        # Validate data
        if not data:
            return jsonify({'error': 'No input data provided.'}), 400
        
        question = data.get('question')
        topic = data.get('topic')
        
        if not question or not topic:
            return jsonify({'error': 'Both "question" and "topic" are required.'}), 400
        
        # Get context for this topic
        full_context = get_topic_content(topic)
        
        # Break into paragraphs for better processing
        paragraphs = re.split(r'\n\n+', full_context)
        
        # Find most relevant paragraphs using semantic search
        sentences = re.split(r'(?<=[.!?])\s+', full_context)
        most_relevant_sentences = semantic_search(question, sentences, top_k=10)
        context = " ".join(most_relevant_sentences[:5])

        
        try:
            # Get the QA pipeline
            qa_pipeline = get_qa_pipeline()
            
            # Run the QA pipeline
            result = qa_pipeline(question=question, context=context)
            answer = result['answer']
            confidence = result['score']
            
            # If confidence is low or answer is too short, enhance the response
            if confidence < 0.5 or len(answer) < 30:
                # Find the paragraph containing the answer for more context
                containing_paragraphs = [p for p in paragraphs if answer in p]
                
                if containing_paragraphs:
                    # Extract surrounding context
                    full_paragraph = containing_paragraphs[0]
                    sentences = re.split(r'(?<=[.!?])\s+', full_paragraph)
                    for i, sentence in enumerate(sentences):
                        if answer in sentence:
                            # Get the sentence and surrounding sentences
                            start_idx = max(0, i-1)
                            end_idx = min(len(sentences), i+2)
                            enhanced_answer = " ".join(sentences[start_idx:end_idx])
                            answer = enhanced_answer
                            break
            
            return jsonify({
                'answer': answer,
                'topic': topic,
                'confidence': confidence
            })
            
        except Exception as e:
            logger.error(f"Error with QA pipeline: {e}")
            # Improved fallback using semantic search
            relevant_sentences = semantic_search(question, sentences, top_k=5)
            
            if relevant_sentences:
                answer = " ".join(relevant_sentences)
            else:
                answer = f"I don't have enough information to answer that question about {topic}."
            
            return jsonify({
                'answer': answer,
                'topic': topic,
                'confidence': 0.3  # Low confidence for fallback
            })
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        return jsonify({'error': str(e)}), 500

@tutor_bp.route('/explain_concept', methods=['POST'])
def explain_concept():
    """Improved endpoint to explain a specific concept within a topic"""
    try:
        data = request.get_json()
        
        # Validate data
        if not data:
            return jsonify({'error': 'No input data provided.'}), 400
        
        topic = data.get('topic')
        concept = data.get('concept')
        
        if not topic or not concept:
            return jsonify({'error': 'Both "topic" and "concept" are required.'}), 400
        
        # Get full context for this topic
        full_context = get_topic_content(topic)
        
        # Create a specific question about the concept
        concept_question = f"What is {concept} in the context of {topic}?"
        
        # Find relevant information using semantic search
        sentences = re.split(r'(?<=[.!?])\s+', full_context)
        
        # Two approaches to find relevance
        # 1. Direct semantic search
        most_relevant_sentences = semantic_search(concept_question, sentences, top_k=8)
        
        # 2. Keyword-based search for validation
        concept_terms = concept.lower().split()
        keyword_relevant = [s for s in sentences if all(term in s.lower() for term in concept_terms)]
        
        # Combine results, prioritizing sentences that appear in both
        combined_sentences = list(set(most_relevant_sentences + keyword_relevant[:5]))
        
        if combined_sentences:
            # Sort by relevance to the concept
            concept_embeddings = get_sentence_model().encode([concept])
            sentence_embeddings = get_sentence_model().encode(combined_sentences)
            similarities = cosine_similarity(concept_embeddings, sentence_embeddings)[0]
            
            # Sort sentences by similarity
            sorted_indices = np.argsort(similarities)[::-1]
            sorted_sentences = [combined_sentences[i] for i in sorted_indices]
            
            # Create explanation
            raw_explanation = " ".join(sorted_sentences[:6])  # Use top sentences
            
            try:
                # Try to summarize for a cleaner explanation
                explanation = summarize_text(raw_explanation, max_length=250)
            except:
                explanation = raw_explanation[:300] + "..."
                
            return jsonify({
                'explanation': explanation,
                'topic': topic,
                'concept': concept
            })
        else:
            # If no relevant sentences found
            return jsonify({
                'explanation': f"I couldn't find specific information about '{concept}' in relation to {topic}.",
                'topic': topic,
                'concept': concept
            })
        
    except Exception as e:
        logger.error(f"Error explaining concept: {e}")
        return jsonify({'error': str(e)}), 500

@tutor_bp.route('/quiz', methods=['POST'])
def generate_quiz():
    """Generate meaningful quiz questions based on Wikipedia content"""
    try:
        data = request.get_json()
        
        # Validate data
        if not data or 'topic' not in data:
            return jsonify({'error': 'No topic provided.'}), 400
        
        topic = data.get('topic')
        num_questions = min(int(data.get('num_questions', 3)), 5)  # Limit to 5 questions
        
        # Get Wikipedia content for this topic
        full_content = get_topic_content(topic)
        
        # Break content into paragraphs
        paragraphs = [p for p in re.split(r'\n\n+', full_content) if len(p.split()) > 20]
        
        # Extract meaningful facts from paragraphs
        fact_paragraphs = []
        for p in paragraphs:
            # Look for paragraphs with dates, proper nouns, or significant historical terms
            if (re.search(r'\b(in|during|after|before) \d{4}\b', p) or  # Contains years
                re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', p) or         # Contains proper names
                re.search(r'\b(revolution|king|war|battle|treaty|constitution|assembly|government)\b', p.lower())):  # Historical terms
                fact_paragraphs.append(p)
        
        # If we don't have enough fact paragraphs, use the regular paragraphs
        if len(fact_paragraphs) < num_questions:
            fact_paragraphs = paragraphs
        
        # Shuffle the paragraphs to get variety
        import random
        random.shuffle(fact_paragraphs)
        
        questions = []
        used_sentences = set()  # To avoid duplicate answers
        
        for paragraph in fact_paragraphs[:min(10, len(fact_paragraphs))]:
            if len(questions) >= num_questions:
                break
                
            # Split paragraph into sentences
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            sentences = [s for s in sentences if len(s.split()) > 10]  # Only substantial sentences
            
            if not sentences:
                continue
                
            # Find sentences with significant historical information
            for sentence in sentences:
                if len(questions) >= num_questions:
                    break
                    
                # Skip if we've already used this sentence
                if sentence in used_sentences:
                    continue
                
                # Extract potential key elements from the sentence
                names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', sentence)
                years = re.findall(r'\b\d{4}\b', sentence)
                events = re.findall(r'\b(revolution|battle|uprising|treaty|declaration|constitution|assembly)\b', sentence.lower())
                
                # Generate a question based on what we found
                question_text = None
                
                # Try to generate a specific question type based on content
                if names and len(names) >= 2 and any(name != topic for name in names):
                    # Question about people
                    person = next((name for name in names if name != topic), names[0])
                    question_text = f"What role did {person} play in the {topic}?"
                elif names and years:
                    # Question about a person and date
                    person = names[0]
                    year = years[0]
                    question_text = f"What significant action was taken by {person} in {year}?"
                elif years and events:
                    # Question about an event and date
                    event = events[0].capitalize()
                    year = years[0]
                    question_text = f"What was the significance of the {event} of {year}?"
                elif years:
                    # Question about a date
                    year = years[0]
                    question_text = f"What significant event occurred in {year} during the {topic}?"
                elif names and any(name != topic for name in names):
                    # Question about a person
                    person = next((name for name in names if name != topic), names[0])
                    question_text = f"Why was {person} important in the {topic}?"
                elif events:
                    # Question about an event
                    event = events[0].capitalize()
                    question_text = f"How did the {event} impact the course of the {topic}?"
                
                # If we couldn't generate a specific question, try another sentence
                if not question_text:
                    continue
                
                # Get the context (surrounding sentences)
                sent_idx = sentences.index(sentence)
                start_idx = max(0, sent_idx - 1)
                end_idx = min(len(sentences), sent_idx + 2)
                context = " ".join(sentences[start_idx:end_idx])
                
                # Add the question
                questions.append({
                    'question': question_text,
                    'answer': sentence,
                    'context': context
                })
                
                # Mark this sentence as used
                used_sentences.add(sentence)
        
        # If we still don't have enough questions, add some generic ones
        if len(questions) < num_questions:
            generic_templates = [
                f"What were the major causes of the {topic}?",
                f"How did the {topic} change European politics?",
                f"What were the main phases of the {topic}?",
                f"What role did economic factors play in the {topic}?",
                f"How did the {topic} influence later revolutionary movements?"
            ]
            
            # Get the remaining paragraphs
            remaining = [p for p in paragraphs if p not in fact_paragraphs]
            if not remaining:
                remaining = paragraphs
            
            for template in generic_templates:
                if len(questions) >= num_questions:
                    break
                
                # Find the most relevant paragraph
                model = get_sentence_model()
                question_embedding = model.encode([template])
                paragraph_embeddings = model.encode(remaining[:min(len(remaining), 10)])
                similarities = cosine_similarity(question_embedding, paragraph_embeddings)[0]
                
                best_idx = np.argmax(similarities)
                best_paragraph = remaining[best_idx]
                
                # Extract the most relevant sentence
                sentences = re.split(r'(?<=[.!?])\s+', best_paragraph)
                if sentences:
                    best_sentence = max(sentences, key=len)
                    
                    # Skip if we've already used this sentence
                    if best_sentence in used_sentences:
                        continue
                    
                    # Add the question
                    questions.append({
                        'question': template,
                        'answer': best_sentence,
                        'context': best_paragraph[:500]  # Limit context size
                    })
                    
                    # Mark this sentence as used
                    used_sentences.add(best_sentence)
        
        return jsonify({
            'topic': topic,
            'quiz': questions
        })
        
    except Exception as e:
        logger.error(f"Error generating quiz: {e}")
        return jsonify({'error': str(e)}), 500

# Function to register this blueprint with CORS
def setup_tutor_blueprint(app):
    CORS(app)  # Enable CORS for all routes
    app.register_blueprint(tutor_bp)