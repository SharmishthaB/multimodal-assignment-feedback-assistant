import express from 'express'
import cors from 'cors'
import dotenv from 'dotenv'
import { createClient } from '@supabase/supabase-js'

dotenv.config()

const app = express()
const PORT = process.env.PORT || 3000

// Middleware
app.use(cors())
app.use(express.json())

// Supabase client
const supabase = createClient(
  process.env.SUPABASE_URL!,
  process.env.SUPABASE_ANON_KEY!
)

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() })
})

// Auth routes (placeholder)
app.post('/api/auth/login', async (req, res) => {
  try {
    const { email, password } = req.body
    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password,
    })
    if (error) return res.status(401).json({ error: error.message })
    res.json(data)
  } catch (err) {
    res.status(500).json({ error: 'Internal server error' })
  }
})

app.post('/api/auth/signup', async (req, res) => {
  try {
    const { email, password } = req.body
    const { data, error } = await supabase.auth.signUp({
      email,
      password,
    })
    if (error) return res.status(400).json({ error: error.message })
    res.json(data)
  } catch (err) {
    res.status(500).json({ error: 'Internal server error' })
  }
})

// Assignments routes (placeholder - will implement in M3)
app.get('/api/assignments', (req, res) => {
  res.json({ message: 'GET assignments - coming in M3' })
})

app.post('/api/assignments', (req, res) => {
  res.json({ message: 'POST assignment - coming in M3' })
})

// Start server
app.listen(PORT, () => {
  console.log(`✅ Server running on http://localhost:${PORT}`)
})
