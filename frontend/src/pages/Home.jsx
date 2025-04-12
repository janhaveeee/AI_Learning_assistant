function Home() {
  return (
    <div className="home-container">
      <h1 className="home-title">
        Welcome to <span className="highlight">SkillMentor 🚀</span>
      </h1>
      <p className="home-subtitle">
        Your smart study companion. Level up your learning journey with ease.
      </p>

      <div className="home-actions">
        <button className="home-button">Get Started</button>
        <button className="home-button secondary">Learn More</button>
      </div>

      <div className="home-features">
        <div className="feature-card">
          <h3>📊 Track Progress</h3>
          <p>Visualize your growth and stay motivated every day.</p>
        </div>
        <div className="feature-card">
          <h3>🧩 Smart Recommendations</h3>
          <p>Personalized learning paths tailored for your success.</p>
        </div>
        <div className="feature-card">
          <h3>🌟 Skill Mastery</h3>
          <p>Sharpen your skills and unlock your potential.</p>
        </div>
      </div>
    </div>
  );
}

export default Home;
