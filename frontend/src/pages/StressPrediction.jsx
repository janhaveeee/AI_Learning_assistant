import { useState } from "react";

export default function StressPrediction() {
  const [form, setForm] = useState({
    time_spent: "",
    retry_attempts: "",
    avg_past_quiz_score: "",
    study_hours_per_week: "",
    mistakes_made: "",
    quizzes_attempted: "",
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      const response = await fetch("http://localhost:5000/predict-stress", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          time_spent: parseFloat(form.time_spent),
          retry_attempts: parseInt(form.retry_attempts),
          avg_past_quiz_score: parseFloat(form.avg_past_quiz_score),
          study_hours_per_week: parseFloat(form.study_hours_per_week),
          mistakes_made: parseInt(form.mistakes_made),
          quizzes_attempted: parseInt(form.quizzes_attempted),
        }),
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      setResult({ error: "Failed to connect to backend." });
    }

    setLoading(false);
  };

  const getBarColor = (level) => {
    switch (level) {
      case "Low":
        return "#4caf50"; // Green
      case "Moderate":
        return "#ffeb3b"; // Yellow
      case "High":
        return "#f44336"; // Red
      default:
        return "#ccc"; // Gray fallback
    }
  };

  return (
    <div className="max-w-xl mx-auto mt-12 px-6 py-8 bg-white shadow-2xl rounded-2xl">
      <h2 className="text-2xl font-bold mb-6 text-center text-gray-800">Predict Stress Level</h2>
      <form onSubmit={handleSubmit} className="space-y-6">
        {[ 
          { label: "Time Spent (hours)", name: "time_spent" },
          { label: "Retry Attempts", name: "retry_attempts" },
          { label: "Avg Past Quiz Score", name: "avg_past_quiz_score", step: "0.1" },
          { label: "Study Hours Per Week", name: "study_hours_per_week" },
          { label: "Mistakes Made", name: "mistakes_made" },
          { label: "Quizzes Attempted", name: "quizzes_attempted" },
        ].map(({ label, name, step }) => (
          <div key={name}>
            <label className="block text-gray-700 mb-1">{label}</label>
            <input
              type="number"
              name={name}
              step={step || "1"}
              value={form[name]}
              onChange={handleChange}
              className="w-full p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-400"
              required
            />
          </div>
        ))}

        <button
          type="submit"
          disabled={loading}
          className="w-full bg-blue-600 text-white font-semibold py-3 rounded-lg hover:bg-blue-700 transition"
        >
          {loading ? "Predicting..." : "Predict Stress"}
        </button>
      </form>

      {result && (
        <div className="mt-6 text-center text-lg font-medium text-green-600">
          {result.error ? (
            result.error
          ) : (
            <>
              <p>📊 Predicted Stress Score: {result.numeric_value.toFixed(2)}</p>
              <p>
                {result.stress_level === "Low" && "😊 Low stress. Keep it up!"}
                {result.stress_level === "Moderate" && "😐 Moderate stress. Try to balance your activities."}
                {result.stress_level === "High" && "😣 High stress detected. Consider relaxing activities!"}
              </p>

              {/* Visual stress level bar */}
              <div
                style={{
                  marginTop: "1rem",
                  width: "100%",
                  height: "20px",
                  backgroundColor: "#eee",
                  borderRadius: "10px",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    width: "100%",
                    height: "100%",
                    backgroundColor: getBarColor(result.stress_level),
                    transition: "background-color 0.5s",
                  }}
                />
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
