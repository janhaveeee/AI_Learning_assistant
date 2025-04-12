import { useState } from "react";
import axios from "axios";

function Predict() {
  const [formData, setFormData] = useState({
    time_spent: "",
    retry_attempts: "",
    videos_watched: "",
    articles_read: "",
    quizzes_attempted: "",
    interactive_exercises: "",
    subject: "",
  });

  const [subjects] = useState(["Math", "Science", "History", "Programming"]);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setPrediction(null);
    setLoading(true);

    try {
      const response = await axios.post("http://localhost:5000/predict", formData);
      setPrediction({
        proficiency: response.data.proficiency,
        probability: response.data.probability,
      });
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

  return (
    <div className="max-w-xl mx-auto mt-10 p-6 bg-white shadow-xl rounded-2xl">
      <h1 className="text-2xl font-bold mb-6 text-center text-blue-600">Predict Proficiency</h1>
      <form onSubmit={handleSubmit} className="space-y-4">
        {Object.keys(formData).map((key) =>
          key !== "subject" ? (
            <div key={key}>
              <label className="block font-medium capitalize text-gray-700">
                {key.replace("_", " ")}:
              </label>
              <input
                type="number"
                name={key}
                value={formData[key]}
                onChange={handleChange}
                required
                className="w-full p-2 mt-1 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-400"
              />
            </div>
          ) : (
            <div key={key}>
              <label className="block font-medium text-gray-700">Subject:</label>
              <select
                name="subject"
                value={formData.subject}
                onChange={handleChange}
                required
                className="w-full p-2 mt-1 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-400"
              >
                <option value="">Select subject</option>
                {subjects.map((subject) => (
                  <option key={subject} value={subject}>
                    {subject}
                  </option>
                ))}
              </select>
            </div>
          )
        )}

        <button
          type="submit"
          className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 rounded-md transition duration-200"
          disabled={loading}
        >
          {loading ? "Predicting..." : "Predict"}
        </button>
      </form>

      {error && <p className="mt-4 text-red-500 font-medium">{error}</p>}

      {prediction && (
        <div className="mt-6 p-4 bg-green-100 rounded-md">
          <p className="font-semibold text-green-800">
            Proficiency:{" "}
            <span className="font-bold">
              {prediction.proficiency === 1 ? "Weak" : "Strong"}
            </span>
          </p>
          <p className="text-green-700">
            Confidence: {(prediction.probability * 100).toFixed(2)}%
          </p>
        </div>
      )}
    </div>
  );
}

export default Predict;
