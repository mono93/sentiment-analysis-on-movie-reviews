import axios from "axios";
import { useState } from "react";

const emojiMap = {
  happy: "😀",
  sad: "😢",
  neutral: "😐",
  angry: "😡",
  default: "",
};

export default function SentimentAnalyzer() {
  const [text, setText] = useState("");
  const [result, setResult] = useState({ prediction: "", confidence: "" });
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (
    e: React.FormEvent<HTMLFormElement>
  ): Promise<void> => {
    e.preventDefault();
    if (!text.trim()) return;

    setLoading(true);
    try {
      const response = await axios.post("http://localhost:8000/analyze", {
        text,
      });
      await new Promise((resolve) => setTimeout(resolve, 500));
      setResult(response.data);
    } catch (error) {
      console.error("Error fetching prediction:", error);
    }
    setLoading(false);
  };

  return (
    <div className="flex flex-col items-center p-6 max-w-lg mx-auto bg-white rounded-2xl shadow-md">
      <h1 className="text-2xl font-bold mb-4 text-gray-800">
        🎭 Sentiment & Emotion Analyzer
      </h1>

      <form onSubmit={handleSubmit} className="w-full">
        <textarea
          className="w-full p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          rows={4}
          placeholder="Type your text here..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <button
          type="submit"
          disabled={loading}
          className="w-full mt-4 py-2 px-4 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50"
        >
          {loading ? "Analyzing..." : "Analyze"}
        </button>
      </form>

      {result && (
        <div className="mt-6 p-4 w-full bg-gray-100 rounded-lg text-center">
          <p className="text-xl font-semibold text-gray-800">
            {emojiMap[result.prediction as keyof typeof emojiMap] ?? emojiMap["default"]}{" "}
            {result.prediction.toUpperCase()}
          </p>
          <p className="text-gray-600">Confidence: {result.confidence}</p>
        </div>
      )}
    </div>
  );
}
