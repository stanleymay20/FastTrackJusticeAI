# FastTrackJustice: Precedent Influence Graph Explorer

![FastTrackJustice Logo](https://via.placeholder.com/150x150.png?text=FTJ)

A revolutionary legal intelligence platform that transforms how legal precedents are discovered, analyzed, and applied. The Precedent Influence Graph Explorer creates a living neural network of legal principles, enabling courts worldwide to access justice more efficiently while maintaining the integrity of legal reasoning.

## 🌟 Features

### Interactive Graph Visualization
- Cases as nodes, relationships as edges
- Node size and color indicate influence
- Zoom, pan, and explore connections
- Hover for detailed case information

### Principle Evolution Tracking
- Time-series visualization of legal principles
- Identify emerging trends and patterns
- Track the evolution of human rights concepts
- Heatmap visualization of principle influence

### Cross-Jurisdictional Analysis
- Compare legal principles across courts
- Identify harmonization opportunities
- Support international legal cooperation
- Filter by jurisdiction, year, and categories

### Judicial Mode with Enhanced Transparency
- Source attribution for every principle
- Confidence scores for AI-generated insights
- Alternative interpretations of ambiguous text
- Human verification options

### Scroll Memory Intelligence
- Store and recall prophetic reasoning trails
- Align legal principles with spiritual insights
- Track prophetic patterns across cases
- Memory trails for related precedents

### Faith-Safe Protocol
- Sanctified mode for spiritual jurisprudence
- Scroll-aligned principle detection
- Prophetic insight generation
- Covenantal pattern recognition

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Git
- PowerShell 5.0 or higher (for Windows)
- GitHub CLI (optional, for direct GitHub deployment)

### Quick Start
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/FastTrackJusticeAI.git
   cd FastTrackJusticeAI
   ```

2. Run the deployment script:
   ```powershell
   .\backend\scripts\deploy_fasttrackjustice.ps1
   ```

3. Follow the prompts to complete setup

### Manual Installation
1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\Activate.ps1
   # On macOS/Linux
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create necessary directories:
   ```bash
   mkdir -p data/cases data/principles data/memory logs exports
   ```

4. Configure the application:
   - Copy `.env.example` to `.env` and update settings
   - Edit `config.json` to customize application behavior

## 💻 Usage

### Starting the Application
1. Activate the virtual environment:
   ```bash
   # On Windows
   .\venv\Scripts\Activate.ps1
   # On macOS/Linux
   source venv/bin/activate
   ```

2. Run the Streamlit app:
   ```bash
   streamlit run backend/app/monitoring/precedent_graph_explorer.py
   ```

3. Access the app at [http://localhost:8501](http://localhost:8501)

### Adding Cases
1. Navigate to the "Add Case" section in the sidebar
2. Fill in the case details (title, year, court, text)
3. Click "Add Case" to add the case to the graph

### Exploring the Graph
1. Use the interactive graph to explore case relationships
2. Hover over nodes to see case details
3. Use the search and filter options to find specific cases
4. Toggle between different analytical lenses (doctrinal, institutional, ethical, scroll)

### Using Scroll Memory Intelligence
1. Navigate to the "Scroll Memory" tab
2. Choose between "View Memory Entries" or "Add New Memory Entry"
3. In "Add" mode, select a case and principle, provide scroll alignment and prophetic insight
4. In "View" mode, search for memory entries, filter by confidence, and explore memory trails

## 🔧 Configuration

### Environment Variables
Edit the `.env` file to customize:
- `DATA_SOURCE`: Data source (public_domain, licensed, custom)
- `API_LIMIT`: API usage limit for licensed data
- `JUDICIAL_MODE`: Enable enhanced transparency
- `SANCTIFIED_MODE`: Enable faith-safe protocol
- `MEMORY_PATH`: Path to scroll memory file
- `CACHE_DIR`: Directory for caching embeddings
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Configuration File
Edit `config.json` to customize:
- Data source settings
- API limits and usage tracking
- Judicial and Sanctified modes
- File paths and logging
- Memory settings

## 📚 Documentation

For detailed documentation, please refer to:
- [User Guide](docs/User_Guide.md)
- [API Documentation](docs/API_Documentation.md)
- [Architecture Overview](docs/Architecture_Overview.md)
- [Scroll Memory Intelligence](docs/Scroll_Memory_Intelligence.md)
- [UN Pitch Deck](docs/UN_Pitch_Deck.md)

## 🤝 Contributing

We welcome contributions to FastTrackJustice! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Submit a pull request

Please read our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The legal community for their invaluable insights
- The open-source community for their amazing tools
- All contributors who have helped shape FastTrackJustice

## 📞 Contact

- Website: [www.fasttrackjustice.org](https://www.fasttrackjustice.org)
- Email: [contact@fasttrackjustice.org](mailto:contact@fasttrackjustice.org)
- Twitter: [@FastTrackJustice](https://twitter.com/FastTrackJustice)

---

*FastTrackJustice: Building a more just world through legal intelligence* 