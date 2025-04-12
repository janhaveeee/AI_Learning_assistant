import { useState } from "react";
import axios from "axios";
import "./Tutor.css"; // We'll need to create this CSS file separately

function Tutor() {
  const [topic, setTopic] = useState("");
  const [question, setQuestion] = useState("");
  const [concept, setConcept] = useState("");
  const [difficultyLevel, setDifficultyLevel] = useState("beginner");
  const [activeTab, setActiveTab] = useState("ask");
  const [response, setResponse] = useState(null);
  const [introduction, setIntroduction] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [quizQuestions, setQuizQuestions] = useState([]);
  const [showAnswers, setShowAnswers] = useState({});

  // Initialize API client
  const api = axios.create({
    baseURL: "http://localhost:5000",
    headers: {
      "Content-Type": "application/json",
    },
  });

  // Set a new topic for tutoring
  const handleSetTopic = async (e) => {
    e.preventDefault();
    setError("");
    setResponse(null);
    setIntroduction("");
    setLoading(true);

    try {
      const result = await api.post("/set_topic", {
        topic: topic
      });

      setIntroduction(result.data.introduction);
    } catch (err) {
      if (err.response?.data?.error) {
        setError(err.response.data.error);
      } else {
        setError("An error occurred while setting the topic. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  // Ask a question about the current topic
  const handleAskQuestion = async (e) => {
    e.preventDefault();
    setError("");
    setResponse(null);
    setLoading(true);

    try {
      const result = await api.post("/ask_question", {
        topic: topic,
        question: question,
        level: difficultyLevel
      });

      setResponse(result.data.answer);
    } catch (err) {
      if (err.response?.data?.error) {
        setError(err.response.data.error);
      } else {
        setError("An error occurred. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  // Request an explanation of a specific concept
  const handleExplainConcept = async (e) => {
    e.preventDefault();
    setError("");
    setResponse(null);
    setLoading(true);

    try {
      const result = await api.post("/explain_concept", {
        topic: topic,
        concept: concept,
        level: difficultyLevel
      });

      setResponse(result.data.explanation);
    } catch (err) {
      if (err.response?.data?.error) {
        setError(err.response.data.error);
      } else {
        setError("An error occurred. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  // Generate a quiz about the current topic
  const handleGenerateQuiz = async (e) => {
    e.preventDefault();
    setError("");
    setQuizQuestions([]);
    setShowAnswers({});
    setLoading(true);

    try {
      const result = await api.post("/quiz", {
        topic: topic,
        num_questions: 3
      });

      setQuizQuestions(result.data.quiz);
    } catch (err) {
      if (err.response?.data?.error) {
        setError(err.response.data.error);
      } else {
        setError("An error occurred generating the quiz. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  // Toggle showing an answer for a specific quiz question
  const toggleAnswer = (index) => {
    setShowAnswers(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  return (
    <div className="tutor-container">
      <h1>AI Tutor Assistant</h1>
      
      {/* Topic Selection */}
      <div className="topic-section">
        <h2>Select a Topic</h2>
        <form onSubmit={handleSetTopic}>
          <div className="form-group">
            <label>Topic:</label>
            <input
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              required
              placeholder="e.g., Quantum Physics, French Revolution, Machine Learning"
            />
          </div>
          <button type="submit" disabled={loading || !topic}>
            {loading ? "Loading..." : "Set Topic"}
          </button>
        </form>
      </div>

      {/* Introduction Display */}
      {introduction && (
        <div className="introduction">
          <h3>Introduction</h3>
          <p>{introduction}</p>
        </div>
      )}

      {/* Tab Navigation */}
      {topic && introduction && (
        <div className="tabs">
          <button 
            className={activeTab === "ask" ? "active" : ""} 
            onClick={() => setActiveTab("ask")}
          >
            Ask Question
          </button>
          <button 
            className={activeTab === "explain" ? "active" : ""} 
            onClick={() => setActiveTab("explain")}
          >
            Explain Concept
          </button>
          <button 
            className={activeTab === "quiz" ? "active" : ""} 
            onClick={() => setActiveTab("quiz")}
          >
            Take Quiz
          </button>
        </div>
      )}

      {/* Difficulty Level Selector */}
      {topic && introduction && activeTab !== "quiz" && (
        <div className="difficulty-selector">
          <label>Difficulty Level:</label>
          <select 
            value={difficultyLevel}
            onChange={(e) => setDifficultyLevel(e.target.value)}
          >
            <option value="beginner">Beginner</option>
            <option value="intermediate">Intermediate</option>
            <option value="advanced">Advanced</option>
          </select>
        </div>
      )}

      {/* Ask Question Tab */}
      {topic && introduction && activeTab === "ask" && (
        <div className="tab-content">
          <h3>Ask a Question</h3>
          <form onSubmit={handleAskQuestion}>
            <div className="form-group">
              <input
                type="text"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                required
                placeholder="Type your question here..."
              />
            </div>
            <button type="submit" disabled={loading || !question}>
              {loading ? "Thinking..." : "Ask"}
            </button>
          </form>
        </div>
      )}

      {/* Explain Concept Tab */}
      {topic && introduction && activeTab === "explain" && (
        <div className="tab-content">
          <h3>Explain a Concept</h3>
          <form onSubmit={handleExplainConcept}>
            <div className="form-group">
              <input
                type="text"
                value={concept}
                onChange={(e) => setConcept(e.target.value)}
                required
                placeholder="Enter the concept you want explained..."
              />
            </div>
            <button type="submit" disabled={loading || !concept}>
              {loading ? "Generating Explanation..." : "Explain"}
            </button>
          </form>
        </div>
      )}

      {/* Quiz Tab */}
      {topic && introduction && activeTab === "quiz" && (
        <div className="tab-content">
          <h3>Quiz on {topic}</h3>
          {quizQuestions.length === 0 ? (
            <div>
              <p>Generate a quiz to test your knowledge on this topic.</p>
              <button onClick={handleGenerateQuiz} disabled={loading}>
                {loading ? "Generating Quiz..." : "Generate Quiz"}
              </button>
            </div>
          ) : (
            <div className="quiz-questions">
              {quizQuestions.map((q, index) => (
                <div key={index} className="quiz-question">
                  <p><strong>Question {index + 1}:</strong> {q.question}</p>
                  <button 
                    onClick={() => toggleAnswer(index)} 
                    className="toggle-answer"
                  >
                    {showAnswers[index] ? "Hide Answer" : "Show Answer"}
                  </button>
                  {showAnswers[index] && (
                    <p className="answer"><strong>Answer:</strong> {q.answer}</p>
                  )}
                </div>
              ))}
              <button onClick={handleGenerateQuiz} disabled={loading}>
                Generate New Quiz
              </button>
            </div>
          )}
        </div>
      )}

      {/* Response Display */}
      {response && (
        <div className="response">
          <h3>Response</h3>
          <p>{response}</p>
        </div>
      )}

      {/* Error Display */}
      {error && <p className="error">{error}</p>}
    </div>
  );
}

export default Tutor;