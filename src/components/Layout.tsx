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
  X
} from 'lucide-react'
import { useState } from 'react'

const Layout: React.FC = () => {
  const { signOut, userProfile } = useAuth()
  const location = useLocation()
  const navigate = useNavigate()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

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
    </div>
  )
}

export default Layout