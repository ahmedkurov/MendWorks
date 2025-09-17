import React from 'react'
import { Outlet, Link, useLocation, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { 
  Activity, 
  Settings, 
  LogOut, 
  Home,
  Plus,
  User,
  Menu,
  X,
  Bot
} from 'lucide-react'
import { useState } from 'react'

const Layout: React.FC = () => {
  const { signOut, userProfile } = useAuth()
  const location = useLocation()
  const navigate = useNavigate()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [chatbotOpen, setChatbotOpen] = useState(false)

  const handleSignOut = async () => {
    try {
      await signOut()
      navigate('/')
    } catch (error) {
      console.error('Error signing out:', error)
    }
  }

  const navigation = [
    { name: 'Dashboard', href: '/dashboard', icon: Home },
    { name: 'Add Device', href: '/add-device', icon: Plus },
    { name: 'Settings', href: '/settings', icon: Settings },
  ]

  const isActive = (path: string) => location.pathname === path

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Mobile header */}
      <div className="md:hidden bg-white shadow-sm border-b">
        <div className="px-4 py-3 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Activity className="w-8 h-8 text-teal-600" />
            <h1 className="text-xl font-bold text-gray-900">MendWorks</h1>
          </div>
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="p-2 text-gray-600 hover:text-gray-900"
          >
            {mobileMenuOpen ? (
              <X className="w-6 h-6" />
            ) : (
              <Menu className="w-6 h-6" />
            )}
          </button>
        </div>
        
        {/* Mobile menu */}
        {mobileMenuOpen && (
          <div className="border-t bg-white">
            <div className="px-4 py-2">
              {navigation.map((item) => {
                const Icon = item.icon
                return (
                  <Link
                    key={item.name}
                    to={item.href}
                    onClick={() => setMobileMenuOpen(false)}
                    className={`flex items-center space-x-3 px-3 py-3 rounded-lg text-sm font-medium ${
                      isActive(item.href)
                        ? 'bg-teal-50 text-teal-700 border-l-4 border-teal-600'
                        : 'text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    <Icon className="w-5 h-5" />
                    <span>{item.name}</span>
                  </Link>
                )
              })}
              
              {/* AI Chatbot Button - Mobile */}
              <button
                onClick={() => {
                  setChatbotOpen(true)
                  setMobileMenuOpen(false)
                }}
                className="flex items-center space-x-3 px-3 py-3 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50 w-full"
              >
                <Bot className="w-5 h-5" />
                <span>AI Assistant</span>
              </button>
              
              <div className="border-t mt-4 pt-4">
                <div className="flex items-center space-x-3 px-3 py-2 text-sm text-gray-600">
                  <User className="w-5 h-5" />
                  <span>{userProfile?.username || 'Technician'}</span>
                </div>
                <button
                  onClick={handleSignOut}
                  className="flex items-center space-x-3 px-3 py-3 rounded-lg text-sm font-medium text-red-600 hover:bg-red-50 w-full"
                >
                  <LogOut className="w-5 h-5" />
                  <span>Sign Out</span>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="hidden md:flex h-screen">
        {/* Desktop sidebar */}
        <div className="w-64 bg-white shadow-lg border-r">
          <div className="p-6 border-b">
            <div className="flex items-center space-x-3">
              <Activity className="w-8 h-8 text-teal-600" />
              <h1 className="text-xl font-bold text-gray-900">MendWorks</h1>
            </div>
          </div>

          <nav className="p-4 space-y-2">
            {navigation.map((item) => {
              const Icon = item.icon
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={`flex items-center space-x-3 px-4 py-3 rounded-lg text-sm font-medium transition-colors ${
                    isActive(item.href)
                      ? 'bg-teal-50 text-teal-700 border-l-4 border-teal-600'
                      : 'text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span>{item.name}</span>
                </Link>
              )
            })}
            
            {/* AI Chatbot Button */}
            <button
              onClick={() => setChatbotOpen(true)}
              className="flex items-center space-x-3 px-4 py-3 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors w-full"
            >
              <Bot className="w-5 h-5" />
              <span>AI Assistant</span>
            </button>
          </nav>

          <div className="absolute bottom-0 w-64 p-4 border-t bg-white">
            <div className="flex items-center space-x-3 mb-4">
              <div className="w-8 h-8 bg-teal-100 rounded-full flex items-center justify-center">
                <User className="w-5 h-5 text-teal-600" />
              </div>
              <div>
                <p className="text-sm font-medium text-gray-900">{userProfile?.username || 'Technician'}</p>
                <p className="text-xs text-gray-600">{userProfile?.hospitals?.name || 'Hospital'}</p>
              </div>
            </div>
            <button
              onClick={handleSignOut}
              className="flex items-center space-x-2 text-sm text-red-600 hover:text-red-700 transition-colors"
            >
              <LogOut className="w-4 h-4" />
              <span>Sign Out</span>
            </button>
          </div>
        </div>

        {/* Main content */}
        <div className="flex-1 overflow-auto">
          <Outlet />
        </div>
      </div>

      {/* Mobile main content */}
      <div className="md:hidden">
        <Outlet />
      </div>

      {/* AI Chatbot Popup */}
      {chatbotOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl shadow-2xl w-full max-w-md h-[600px] flex flex-col">
            {/* Chatbot Header */}
            <div className="flex items-center justify-between p-4 border-b bg-teal-50 rounded-t-xl">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-teal-100 rounded-full flex items-center justify-center">
                  <Bot className="w-5 h-5 text-teal-600" />
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900">AI Assistant</h3>
                  <p className="text-xs text-gray-600">Medical Device Support</p>
                </div>
              </div>
              <button
                onClick={() => setChatbotOpen(false)}
                className="text-gray-400 hover:text-gray-600 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Chat Messages */}
            <div className="flex-1 p-4 overflow-y-auto space-y-4">
              {/* Welcome Message */}
              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-teal-100 rounded-full flex items-center justify-center flex-shrink-0">
                  <Bot className="w-4 h-4 text-teal-600" />
                </div>
                <div className="bg-gray-100 rounded-lg p-3 max-w-xs">
                  <p className="text-sm text-gray-800">
                    Hello! I'm your AI assistant for medical device maintenance. How can I help you today?
                  </p>
                </div>
              </div>

              {/* Sample Messages */}
              <div className="flex items-start space-x-3 justify-end">
                <div className="bg-teal-600 text-white rounded-lg p-3 max-w-xs">
                  <p className="text-sm">What devices need maintenance?</p>
                </div>
                <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center flex-shrink-0">
                  <User className="w-4 h-4 text-gray-600" />
                </div>
              </div>

              <div className="flex items-start space-x-3">
                <div className="w-8 h-8 bg-teal-100 rounded-full flex items-center justify-center flex-shrink-0">
                  <Bot className="w-4 h-4 text-teal-600" />
                </div>
                <div className="bg-gray-100 rounded-lg p-3 max-w-xs">
                  <p className="text-sm text-gray-800">
                    I can help you check device status, schedule maintenance, and provide troubleshooting guidance. What would you like to know?
                  </p>
                </div>
              </div>

              {/* Quick Actions */}
              <div className="space-y-2">
                <p className="text-xs text-gray-500 font-medium">Quick Actions:</p>
                <div className="flex flex-wrap gap-2">
                  <button className="px-3 py-1 bg-teal-100 text-teal-700 rounded-full text-xs hover:bg-teal-200 transition-colors">
                    Check Device Status
                  </button>
                  <button className="px-3 py-1 bg-teal-100 text-teal-700 rounded-full text-xs hover:bg-teal-200 transition-colors">
                    Schedule Maintenance
                  </button>
                  <button className="px-3 py-1 bg-teal-100 text-teal-700 rounded-full text-xs hover:bg-teal-200 transition-colors">
                    Troubleshooting
                  </button>
                </div>
              </div>
            </div>

            {/* Chat Input */}
            <div className="p-4 border-t">
              <div className="flex space-x-2">
                <input
                  type="text"
                  placeholder="Type your message..."
                  className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent text-sm"
                />
                <button className="px-4 py-2 bg-teal-600 text-white rounded-lg hover:bg-teal-700 transition-colors">
                  <Bot className="w-4 h-4" />
                </button>
              </div>
              <p className="text-xs text-gray-500 mt-2">
                AI Assistant is here to help with device maintenance and troubleshooting.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default Layout