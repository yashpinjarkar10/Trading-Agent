# Trading Agent Frontend

Modern React + Vite frontend for the Trading Agent platform with real-time charts, stock analysis, and AI chat capabilities.

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm/yarn/pnpm

### Installation

1. **Install dependencies**
   ```bash
   npm install
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env if needed (default: http://localhost:8000)
   ```

3. **Start development server**
   ```bash
   npm run dev
   ```

4. **Access the application**
   - Frontend: http://localhost:5173
   - Make sure backend is running on http://localhost:8000

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/      # React components
│   │   ├── ChartView.jsx
│   │   ├── AIAnalyst.jsx
│   │   ├── AnalysisForm.jsx
│   │   ├── ChatInterface.jsx
│   │   └── ResultsDisplay.jsx
│   ├── services/        # API service layer
│   │   └── api.js
│   ├── styles/          # CSS styles
│   │   └── index.css
│   ├── config/          # Configuration
│   │   └── constants.js
│   ├── App.jsx          # Main app component
│   └── main.jsx         # Entry point
├── public/              # Static assets
├── index.html           # HTML template
├── vite.config.js       # Vite configuration
└── package.json         # Dependencies
```

## 🎨 Features

### Chart View
- Real-time TradingView charts
- Symbol search and switching
- Quick analysis buttons

### AI Analyst
- **Direct Analysis Mode**: Run technical, fundamental, and news analysis
- **AI Chat Mode**: Natural language queries about stocks

### Components
- **ChartView**: TradingView integration
- **AIAnalyst**: Mode switcher (Direct/Chat)
- **AnalysisForm**: Multi-analysis form with options
- **ChatInterface**: AI chat with conversation history
- **ResultsDisplay**: Formatted analysis results with tabs

## 🔧 Configuration

### Environment Variables

Create a `.env` file:
```env
VITE_API_URL=http://localhost:8000
```

For production:
```env
VITE_API_URL=https://your-backend-api.com
```

### Vite Configuration

The `vite.config.js` includes:
- React plugin
- Development proxy to backend API
- Build optimization

## 📦 Build for Production

```bash
# Build optimized production bundle
npm run build

# Preview production build locally
npm run preview
```

The build output will be in the `dist/` directory.

## 🚀 Deployment

### Static Hosting (Vercel, Netlify, Cloudflare Pages)

1. **Build the project**
   ```bash
   npm run build
   ```

2. **Deploy the `dist/` folder**
   - Set environment variable: `VITE_API_URL=<your-backend-url>`

### Vercel
```bash
npm install -g vercel
vercel --prod
```

### Netlify
```bash
npm install -g netlify-cli
netlify deploy --prod --dir=dist
```

## 🛠️ Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

### Code Style

The project uses ESLint for code quality. Run:
```bash
npm run lint
```

## 🔌 API Integration

The frontend communicates with the backend via REST API:

- `POST /api/analysis/technical` - Technical analysis
- `POST /api/analysis/fundamental` - Fundamental analysis
- `POST /api/analysis/news` - News sentiment
- `POST /api/chat` - AI chat
- `GET /api/health` - Health check
- `GET /api/tickers` - Popular tickers

See `src/services/api.js` for implementation.

## 🎨 Styling

The application uses a custom dark theme with:
- Glass morphism effects
- Gradient backgrounds
- Responsive design
- Modern UI components

Main styles are in `src/styles/index.css`.

## 📱 Responsive Design

The interface is fully responsive and works on:
- Desktop (1920px+)
- Laptop (1366px+)
- Tablet (768px+)
- Mobile (375px+)

## 🔐 Security

- Environment variables for API URLs
- No sensitive data in frontend code
- CORS handled by backend
- Input validation on forms

## 📄 License

MIT License - See main project README
