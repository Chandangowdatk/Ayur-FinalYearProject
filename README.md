# AyurRAG: Ayurvedic Knowledge Retrieval System

## Project Overview
AyurRAG is an advanced Retrieval-Augmented Generation (RAG) system designed to provide accurate and contextual information about Ayurvedic medicine and practices. The system combines modern NLP techniques with traditional Ayurvedic knowledge to deliver precise responses to user queries.

## Key Features
- **Intelligent Query Processing**: Advanced NLP for understanding Ayurvedic terminology
- **Vector Database Integration**: Efficient similarity search using Milvus
- **Contextual Response Generation**: Accurate and relevant answers based on Ayurvedic texts
- **Interactive User Interface**: User-friendly chat interface
- **Multi-language Support**: Text-to-speech capabilities in multiple languages
- **Secure Authentication**: User management and session handling

## Technical Stack
- **Backend**: Django, Django REST Framework
- **Frontend**: HTML5, CSS3, JavaScript
- **Database**: PostgreSQL, Milvus (Vector Database)
- **AI/ML**: Sentence Transformers, RAG Architecture
- **DevOps**: Docker, GitHub Actions

## Project Structure
```
ayurrag/
├── manage.py
├── requirements.txt
├── ayurrag/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── chat/
│   ├── __init__.py
│   ├── models.py
│   ├── views.py
│   ├── urls.py
│   └── tests.py
├── static/
│   └── css/
│       └── styles.css
└── templates/
    └── chat/
        └── index.html
```

## Setup and Installation

### Prerequisites
- Python 3.12+
- Docker (for Milvus)
- AWS Account (for deployment)
- Git

### Local Development Setup
1. Clone the repository
```bash
git clone https://github.com/Chandangowdatk/AyurRAG.git
cd AyurRAG
```

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up environment variables
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Run migrations
```bash
python manage.py migrate
```

6. Start development server
```bash
python manage.py runserver
```

## Usage
1. Access the application at `http://localhost:8000`
2. Create an account or login
3. Start asking questions about Ayurvedic medicine and practices
4. Use the text-to-speech feature for audio responses

## API Documentation
The API documentation is available at `/api/docs/` when running the server locally.

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Contributors
- [Chandan Gowda T K](https://github.com/Chandangowdatk)
- [Dhanush H](https://github.com/dealga)
- Srivatsa G
- Shrisha G Shetty

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Ayurvedic texts and resources
- Open source community
- Project mentors and advisors

## Contact
For any queries or support, please open an issue in the repository or contact the project maintainers. 
