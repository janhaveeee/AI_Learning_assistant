import React, { useState } from "react";
import axios from "axios";

const DifficultyPredictor = () => {
  const [formData, setFormData] = useState({
    time_spent: "",
    retry_attempts: "",
    videos_watched: "",
    articles_read: "",
    quizzes_attempted: "",
    interactive_exercises: "",
    subject: "",
  });

  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const interpretResult = (level) => {
    switch (level) {
      case 0:
        return "Strong (Easy to Understand)";
      case 1:
        return "Weak (Needs Improvement)";
      default:
        return "Unknown";
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setResult(null);

    try {
      const response = await axios.post("http://localhost:5000/predict_difficulty_level", formData);
      setResult(response.data.difficulty_level);
    } catch (err) {
      setError(err.response?.data?.error || "Something went wrong");
    }
  };

  return (
    <div className="p-6 max-w-md mx-auto bg-white rounded-xl shadow-md space-y-4">
      <h2 className="text-xl font-bold text-center">Predict Difficulty Level</h2>
      <form onSubmit={handleSubmit} className="space-y-3">
        {Object.keys(formData).map((key) => (
          <div key={key}>
            <label className="block mb-1 capitalize">{key.replaceAll("_", " ")}</label>
            <input
              type={key === "subject" ? "text" : "number"}
              name={key}
              value={formData[key]}
              onChange={handleChange}
              required
              className="w-full border border-gray-300 rounded px-3 py-2"
            />
          </div>
        ))}

        <button type="submit" className="w-full bg-blue-500 text-white py-2 rounded">
          Predict
        </button>
      </form>

      {result !== null && (
        <div className={`mt-4 font-semibold ${result === 0 ? "text-green-600" : "text-red-600"}`}>
          Predicted Difficulty Level: {interpretResult(result)}
        </div>
      )}

      {error && (
        <div className="mt-4 text-red-600">
          Error: {error}
        </div>
      )}
    </div>
  );
};

export default DifficultyPredictor;
