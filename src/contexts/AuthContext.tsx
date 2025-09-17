import React, { createContext, useContext, useEffect, useState } from 'react'
import { User, Session } from '@supabase/supabase-js'
import { supabase } from '../lib/supabase'

const isConfigured = () => {
  return import.meta.env.VITE_SUPABASE_URL && 
         import.meta.env.VITE_SUPABASE_URL !== 'https://placeholder.supabase.co' &&
         import.meta.env.VITE_SUPABASE_ANON_KEY && 
         !import.meta.env.VITE_SUPABASE_ANON_KEY.includes('placeholder')
}

interface AuthContextType {
  user: User | null
  session: Session | null
  loading: boolean
  profileLoading: boolean
  signUp: (email: string, password: string, userData: any) => Promise<any>
  signIn: (email: string, password: string) => Promise<any>
  signOut: () => Promise<void>
  userProfile: any
  refreshUserProfile: () => Promise<void>
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null)
  const [session, setSession] = useState<Session | null>(null)
  const [loading, setLoading] = useState(true)
  const [profileLoading, setProfileLoading] = useState(false)
  const [userProfile, setUserProfile] = useState<any>(null)

  const refreshUserProfile = async () => {
    if (!user) return

    setProfileLoading(true)
    try {
      console.log('Fetching user profile for user:', user.id)
      const startTime = Date.now()
      
      const { data, error } = await supabase
        .from('users')
        .select(`
          *,
          hospitals (*)
        `)
        .eq('id', user.id)
        .maybeSingle()

      const endTime = Date.now()
      console.log(`Profile fetch took ${endTime - startTime}ms`)

      if (error) {
        console.error('Profile fetch error:', error)
        throw error
      }
      
      console.log('Profile data received:', data)
      setUserProfile(data)
    } catch (error) {
      console.error('Error fetching user profile:', error)
      // Set a fallback profile to prevent infinite loading
      setUserProfile(null)
    } finally {
      setProfileLoading(false)
    }
  }

  useEffect(() => {
    let timeoutId: NodeJS.Timeout
    
    // Set a timeout to prevent infinite loading
    timeoutId = setTimeout(() => {
      console.warn('Auth loading timeout - setting loading to false')
      setLoading(false)
    }, 10000) // 10 second timeout

    supabase.auth.getSession().then(({ data: { session } }) => {
      clearTimeout(timeoutId)
      setSession(session)
      setUser(session?.user ?? null)
      setLoading(false)
    }).catch((error) => {
      console.error('Error getting session:', error)
      clearTimeout(timeoutId)
      setLoading(false)
    })

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      clearTimeout(timeoutId)
      setSession(session)
      setUser(session?.user ?? null)
      setLoading(false)
    })

    return () => {
      clearTimeout(timeoutId)
      subscription.unsubscribe()
    }
  }, [])

  useEffect(() => {
    if (user) {
      refreshUserProfile()
    } else {
      setUserProfile(null)
    }
  }, [user])

  const signUp = async (email: string, password: string, metadata?: {
    username: string
    contact_info: string
    hospitalName?: string
    hospitalAddress?: string
    hospitalContact?: string
  }) => {
    if (!isConfigured()) {
      throw new Error('Please configure your Supabase project. Click "Connect to Supabase" to set up your database.')
    }

    const { data, error } = await supabase.auth.signUp({
      email,
      password,
    })

    if (error) throw error
    if (!data.user) throw new Error('Failed to create user')

    // Create hospital first if hospital data is provided
    let hospitalId = null
    if (metadata?.hospitalName) {
      const { data: hospital, error: hospitalError } = await supabase
        .from('hospitals')
        .insert({
          name: metadata.hospitalName,
          address: metadata.hospitalAddress!,
          contact_number: metadata.hospitalContact!
        })
        .select()
        .single()

      if (hospitalError) throw hospitalError
      hospitalId = hospital.id
    }

    // Create user profile
    const { error: profileError } = await supabase
      .from('users')
      .insert({
        id: data.user.id,
        username: metadata?.username || '',
        hospital_id: hospitalId,
        contact_info: metadata?.contact_info || '',
      })

    if (profileError) throw profileError

    return data
  }

  const signIn = async (email: string, password: string) => {
    // Check if Supabase is properly configured
    if (!import.meta.env.VITE_SUPABASE_URL || 
        import.meta.env.VITE_SUPABASE_URL === 'https://placeholder.supabase.co' ||
        import.meta.env.VITE_SUPABASE_ANON_KEY === 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InBsYWNlaG9sZGVyIiwicm9sZSI6ImFub24iLCJpYXQiOjE2NDUxOTI4MDAsImV4cCI6MTk2MDc2ODgwMH0.placeholder') {
      throw new Error('Please configure your Supabase project. Click "Connect to Supabase" to set up your database.')
    }
    
    const { data, error } = await supabase.auth.signInWithPassword({ email, password })
    if (error) throw error
    return data
  }

  const signOut = async () => {
    const { error } = await supabase.auth.signOut()
    if (error) throw error
  }

  const value: AuthContextType = {
    user,
    session,
    loading,
    profileLoading,
    signUp,
    signIn,
    signOut,
    userProfile,
    refreshUserProfile,
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}