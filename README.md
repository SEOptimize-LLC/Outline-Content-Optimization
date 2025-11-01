# Blog Post Optimizer - AI-Powered Content Enhancement

A comprehensive Streamlit application that leverages cutting-edge AI models (GPT-4o, Claude 3.5 Sonnet) and ML/NLP packages to optimize blog content for SEO and AI search tools.

## 📦 Complete Package Files

This repository includes:
- **app.py** - Main Streamlit application
- **requirements.txt** - Python dependencies (includes spaCy model download)
- **.python-version** - Specifies Python 3.11 for compatibility
- **README.md** - Comprehensive documentation (this file)
- **PYTHON_VERSION_GUIDE.md** - Important compatibility guide for Python versions
- **QUICKSTART.md** - 10-minute deployment guide
- **TROUBLESHOOTING.md** - Common issues and solutions
- **secrets.toml.example** - Template for API key configuration
- **.streamlit/config.toml** - Optional UI configuration

## ⚠️ Python Version Notice

**Using Python 3.12?** You'll encounter compatibility errors with spaCy. **Two solutions:**

1. **Recommended:** Add `.python-version` file with "3.11" to use Python 3.11
2. **Alternative:** Use the NLTK-based version (see [PYTHON_VERSION_GUIDE.md](PYTHON_VERSION_GUIDE.md))

The `.python-version` file is already included in this repository, so if you clone it, you'll automatically use Python 3.11.

## 🚀 Quick Start

**New to deployment?** Follow [QUICKSTART.md](QUICKSTART.md) for step-by-step instructions.

**Experienced?** Here's the TL;DR:

1. Clone repo
2. Add at least one API key to Streamlit Secrets
3. Deploy to Streamlit Cloud
4. Wait 3-5 minutes for first load
5. Start optimizing!

---

## 🚀 Features

### Tab 1: Outline Optimizer
- **Deep Audience Research**: ICP analysis, pain points, psychological triggers
- **Search Intent Analysis**: Multi-platform search behavior insights
- **AI-Powered Optimization**: Cross-reference with Query Fan-Out analysis
- **Structured Output**: Optimized outlines with 7-12 talking points per section
- **Side-by-side Comparison**: Visual diff of original vs optimized
- **Export Options**: Download as Markdown

### Tab 2: Draft Optimizer

#### Subprocess 1: Keyword Optimization
- **Semantic Analysis**: NLP-powered relevance scoring
- **Smart Placement**: Context-aware keyword integration
- **Natural Integration**: AI-generated paragraphs maintaining tone
- **Variation Detection**: Identifies semantic similarities (>0.8 threshold)

#### Subprocess 2: AI Tool Optimization (10-Item Checklist)
1. Answer-First Introduction
2. Question-Based H2 Headings
3. Semantic Chunks (75-300 words)
4. Answer-Evidence-Context Formula
5. Direct Sentence Structures (Active Voice)
6. Informational Density (+20%)
7. Attributed Claims
8. First-Hand Experience Signals
9. Dedicated FAQ Section
10. Optimized Title & Meta

### Tab 3: Semantic SEO Analyzer (Koray Tuğberk GÜBÜR Framework)

**The most advanced semantic content analyzer based on industry-leading SEO framework**

#### Macro Context Extraction
- **Primary Topic/Central Entity**: Validates and identifies the true central entity
- **Domain/Authority (Source Context)**: Determines brand identity and monetization strategy
- **User's Intent**: Analyzes dominant search intent (Informational, Transactional, Navigational, Investigational)
- **Search Persona**: Identifies demographics, psychographics, and expertise level
- **Main Benefits**: Extracts value propositions connected to source context
- **Entity-Attribute-Value (EAV) Inventory**: Generates 10-15 key attribute-value pairs
- **Intent & Topical Clusters**: Groups core section (monetization) vs. author section (topical authority)
- **Link Hub Potential**: Recommends ideal root document H1 for semantic content network

#### Micro Context Extraction
- **Semantically Relevant Entities**: Identifies 10-15 supporting entities
- **Key Attributes & Values**: Extracts characteristics with specific data points
- **Predicates**: Analyzes relationships connecting entities and attributes
- **Temporal Elements**: Identifies time-sensitive information, dates, durations
- **Conditional Synonyms**: Finds phrases using conjunctive words (and, or)
- **Co-occurring Terms**: Discovers 5-7 term clusters for distributional semantics
- **Annotation Text Patterns**: Suggests text for internal link context
- **Anchor Segments**: Recommends mutual words for discourse flow
- **Question Types**: Categorizes boolean, definitional, grouping, comparative, temporal questions
- **Modality & Measurement Units**: Extracts scientific terminology and units
- **Macro vs Micro Boundary**: Identifies transition from main to supplementary content

#### Distributional Semantics Analysis
- **Term Frequency Analysis**: Top 20 significant terms with occurrence counts
- **Co-occurrence Matrix**: Identifies term pairs that appear together in sentences
- **Contextual Dance Mapping**: Reveals natural semantic relationships
- **Statistical Linguistics**: Quantitative analysis of term distributions

#### Content Brief Generation (4-Column Framework)
1. **Contextual Vector (The Flow)**: Optimal heading sequence with reasoning
2. **Contextual Hierarchy (The Weight)**: Coverage allocation (60-70% macro, 30-40% micro)
3. **Contextual Structure (The Format)**: Article methodology rules and format specifications
4. **Contextual Connections (Internal Linking)**: Strategic link opportunities with anchor text

#### Comprehensive Reporting
- **Entity Validation Report**: Confirms primary entity accuracy
- **Complete Semantic Analysis**: Macro + Micro context in structured format
- **Distributional Semantics Tables**: Visual co-occurrence data
- **Actionable Content Brief**: Ready-to-implement optimization roadmap
- **Downloadable Reports**: Markdown export for all analyses

## 📋 Requirements

- **Python 3.11** (recommended) or Python 3.12 with modifications
- At least one AI provider API key:
  - **OpenAI API Key** (for GPT-4o models)
  - **Anthropic API Key** (for Claude models)
  - **Google API Key** (for Gemini models)
- Streamlit Cloud account (for deployment)

**⚠️ Important:** If using Python 3.12, see [PYTHON_VERSION_GUIDE.md](PYTHON_VERSION_GUIDE.md) for compatibility instructions. Python 3.11 is strongly recommended for easiest setup.

**Note:** You don't need all three API keys! The app intelligently detects which keys you've configured and only shows models from those providers.

## 🛠️ Local Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd blog-optimizer

# Install dependencies
pip install -r requirements.txt

# Note: The spaCy model will be downloaded automatically via requirements.txt

# Run locally
streamlit run app.py
```

**Important:** The `requirements.txt` includes a direct URL to download the spaCy model. This is required for Streamlit Cloud deployment.

## ☁️ Streamlit Cloud Deployment

### Step 1: Prepare Your Repository

1. Create a new GitHub repository
2. Add these files:
   - `app.py` (main application)
   - `requirements.txt`
   - `README.md`

### Step 2: Configure Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub repository
4. Select the branch and `app.py` as the main file

### Step 3: Add API Keys to Secrets

In Streamlit Cloud:
1. Go to your app settings
2. Navigate to "Secrets" section
3. Add your API keys (one or more):

**Option 1: Use all three (recommended for flexibility)**
```toml
OPENAI_API_KEY = "sk-your-openai-api-key-here"
ANTHROPIC_API_KEY = "sk-ant-your-anthropic-key-here"
GOOGLE_API_KEY = "AIzaSy-your-google-key-here"
```

**Option 2: Use only one provider**
```toml
# Just add the one(s) you have:
GOOGLE_API_KEY = "AIzaSy-your-google-key-here"
```

4. Save and deploy

**The app will automatically detect which APIs you've configured and only show those models!**

### Step 4: Deploy

Click "Deploy!" and wait for the app to build. First deployment may take 3-5 minutes.

## 🔐 Security Best Practices

- **Never commit API keys** to your repository
- **Always use Streamlit Secrets** for sensitive data
- **Rotate keys regularly** for enhanced security
- **Monitor API usage** to prevent unexpected charges

## 💡 Usage Guide

### Outline Optimizer Workflow

1. **Enter Primary Keyword**: Your target topic or keyword
2. **Paste Draft Outline**: Use Markdown format with H2/H3 structure
   ```markdown
   ## Introduction
   ## Main Topic 1
   ### Subtopic 1.1
   ### Subtopic 1.2
   ## Conclusion
   ```
3. **Add Query Fan-Out Report** (optional): Upload or paste analysis
4. **Click "Optimize Outline"**: AI performs:
   - Audience research (demographics, pain points, intent)
   - Cross-reference analysis
   - Outline enhancement with talking points
5. **Review Results**: Side-by-side comparison
6. **Download**: Export optimized outline as Markdown

### Draft Optimizer Workflow

1. **Enter Primary Keyword**: Your target SEO keyword
2. **Paste Content Draft**: Full blog post (Markdown/HTML)
3. **Add Keyword List**: Line-separated or comma-separated keywords
   ```
   dog training methods
   puppy obedience
   canine behavior
   ```
4. **Click "Optimize Draft"**: Two-stage process:

   **Stage 1 - Keyword Optimization:**
   - Relevance scoring for each keyword
   - Placement suggestions with context
   - AI-generated integration snippets

   **Stage 2 - AI Tool Optimization:**
   - 10-item checklist application
   - Full draft rewrite for AI search visibility
   - Compliance report

5. **Review Optimizations**:
   - Keyword integration table
   - Checklist metrics
   - Final optimized draft
6. **Download**: Export as Markdown

### Semantic SEO Analyzer Workflow

1. **Enter Primary Entity**: Can be a single word ("Germany"), phrase ("Kidney Stones"), or complete question ("How Long Do Teeth Pores Stay Open After Whitening")
2. **Paste Content**: Your blog post draft in Markdown format
3. **Select Analysis Options**:
   - ✅ Validate Primary Entity (confirms if your claimed entity is accurate)
   - ✅ Include Distributional Semantics (co-occurrence analysis)
   - ✅ Generate Content Brief (4-column framework)
4. **Click "Analyze Content"**: Multi-stage semantic analysis:

   **Stage 1 - Entity Validation:**
   - Validates claimed primary entity
   - Identifies actual central entity if different
   - Provides confidence level and reasoning

   **Stage 2 - Macro Context Extraction:**
   - Identifies 8 macro context elements
   - Generates EAV (Entity-Attribute-Value) inventory
   - Maps topical clusters (core vs. author sections)
   - Recommends link hub structure

   **Stage 3 - Micro Context Extraction:**
   - Extracts 12+ micro context elements
   - Identifies semantically relevant entities
   - Analyzes predicates and relationships
   - Categorizes question types
   - Maps macro vs. micro content boundaries

   **Stage 4 - Distributional Semantics (Optional):**
   - Analyzes top 20 significant terms
   - Generates co-occurrence matrix
   - Identifies term pairs and contextual dance

   **Stage 5 - Content Brief Generation (Optional):**
   - Contextual Vector: Optimal heading flow
   - Contextual Hierarchy: Coverage weights
   - Contextual Structure: Format specifications
   - Contextual Connections: Internal linking strategy

5. **Review Analysis**:
   - Expandable sections for each analysis stage
   - Visual tables for distributional semantics
   - Complete content brief with actionable recommendations
6. **Download Reports**:
   - Individual reports (entity validation, content brief)
   - Complete semantic SEO analysis report

## 🧠 Technical Architecture

### NLP Stack
- **Sentence Transformers**: Semantic similarity (all-MiniLM-L6-v2)
- **spaCy**: NLP parsing, entity extraction, syntax analysis
- **scikit-learn**: TF-IDF vectorization, cosine similarity

### AI Integration
- **OpenAI GPT-4o**: Advanced reasoning and content generation
- **Anthropic Claude 3.5 Sonnet**: High-quality optimization and analysis
- **Google Gemini**: Cost-effective processing with free tier
- **Smart Model Detection**: Automatically shows only models from configured API providers
- **Model Selection**: Choose specific models (e.g., GPT-4o, Claude 3.5 Sonnet, Gemini 2.0 Flash)
- **Universal Router**: Seamlessly routes requests to the correct AI provider

### Performance Optimizations
- **Model Caching**: `@st.cache_resource` for NLP models
- **Session State**: Prevents resets on file downloads
- **Progress Indicators**: Real-time feedback for long operations
- **Error Handling**: Graceful degradation with informative messages

## 🐛 Troubleshooting

### Common Issues

**"spaCy model not found" error:**
- Ensure your `requirements.txt` includes the spaCy model download URL (last line)
- Redeploy your app after updating requirements.txt

**False "API Connected" status:**
- Check that secrets are properly formatted (no empty values)
- Verify keys are at least 10 characters long
- Remove any placeholder text like "your-key-here"

**App won't start:**
- Add at least one valid API key to Streamlit Secrets
- Reboot app after adding secrets

**For detailed solutions, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - comprehensive guide with all common issues and fixes.

### Common Issues

**"API key not found in secrets"**
- Ensure keys are added to Streamlit Cloud Secrets
- Check spelling: `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`

**"spaCy model not found"**
- First run downloads model automatically
- If persistent, redeploy app

**"Download button resets app"**
- Fixed with `create_download_link()` using base64 encoding
- Uses HTML links instead of Streamlit download buttons

**Slow performance**
- First run loads models (1-2 min one-time setup)
- Subsequent runs use cached models
- API calls vary by provider and load

### API Rate Limits

- **OpenAI**: Monitor usage at platform.openai.com
- **Anthropic**: Check console.anthropic.com
- **Google Gemini**: Free tier: 60 requests/min, view at aistudio.google.com
- Consider implementing caching for repeated requests
- **Pro Tip**: Start with Gemini's free tier for testing!

## 📊 Example Use Cases

### Use Case 1: Blog Outline Creation
**Input**: "How to train a puppy"
**Process**: 
- Audience research reveals first-time dog owners' anxiety
- Query Fan-Out includes "potty training," "crate training," "socialization"
- Optimized outline addresses pain points with actionable steps

### Use Case 2: Existing Draft Enhancement
**Input**: 2000-word draft on "Email Marketing Best Practices"
**Process**:
- Keyword analysis identifies missing terms: "email segmentation," "A/B testing"
- AI optimization converts to question-based headers
- FAQ section added for long-tail queries
- Result: 2500 words, 30% more informational density

## 🔄 Updates and Maintenance

### Updating Dependencies
```bash
pip install --upgrade streamlit anthropic openai sentence-transformers
```

### Model Updates
- Monitor Anthropic/OpenAI/Google for new model releases
- Update model strings in code (`get_available_models()` function):
  - OpenAI: `"gpt-4o"` (or latest)
  - Anthropic: `"claude-3-5-sonnet-20241022"` (or latest)
  - Google: `"gemini-2.0-flash-exp"` (or latest)
- Add new models to the models dictionary for each provider

## 📝 License

This project is provided as-is for educational and commercial use.

## 🤝 Support

For issues, feature requests, or questions:
1. Check this README first
2. Review Streamlit documentation: [docs.streamlit.io](https://docs.streamlit.io)
3. Check API provider docs:
   - [OpenAI API Docs](https://platform.openai.com/docs)
   - [Anthropic API Docs](https://docs.anthropic.com)
   - [Google Gemini API Docs](https://ai.google.dev/docs)

## 🎯 Roadmap

Recent additions:
- [✅] **Semantic SEO Analyzer** - Koray Tuğberk GÜBÜR framework implementation
- [✅] Entity-Attribute-Value (EAV) extraction
- [✅] Distributional semantics analysis
- [✅] 4-column content brief generator

Future enhancements:
- [ ] Topical map visualization tool
- [ ] Semantic content network builder
- [ ] SERP analysis integration for competitive semantic analysis
- [ ] PDF export functionality
- [ ] Real-time SerpAPI integration for live search data
- [ ] Multi-language support
- [ ] Content scoring dashboard
- [ ] Historical version tracking
- [ ] Bulk processing mode

---

**Built with ❤️ using Streamlit, OpenAI, Anthropic, and cutting-edge NLP**