import React, { useState, useEffect } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { supabase } from '../lib/supabase'
import { User, Building, Phone, Mail, Save, ArrowLeft } from 'lucide-react'

const Settings: React.FC = () => {
  const [userData, setUserData] = useState({
    username: '',
    contactInfo: '',
    email: ''
  })
  const [hospitalData, setHospitalData] = useState({
    name: '',
    address: '',
    contactNumber: ''
  })
  const [loading, setLoading] = useState(false)
  const [success, setSuccess] = useState('')
  const [error, setError] = useState('')

  const { user, userProfile, refreshUserProfile } = useAuth()

  useEffect(() => {
    if (userProfile) {
      setUserData({
        username: userProfile.username || '',
        contactInfo: userProfile.contact_info || '',
        email: user?.email || ''
      })
      
      if (userProfile.hospitals) {
        setHospitalData({
          name: userProfile.hospitals.name || '',
          address: userProfile.hospitals.address || '',
          contactNumber: userProfile.hospitals.contact_number || ''
        })
      }
    }
  }, [userProfile, user])

  const handleUserDataChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setUserData(prev => ({
      ...prev,
      [e.target.name]: e.target.value
    }))
  }

  const handleHospitalDataChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setHospitalData(prev => ({
      ...prev,
      [e.target.name]: e.target.value
    }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')
    setSuccess('')

    try {
      // Update user profile
      const { error: userError } = await supabase
        .from('users')
        .update({
          username: userData.username,
          contact_info: userData.contactInfo
        })
        .eq('id', user!.id)

      if (userError) throw userError

      // Update hospital information
      const { error: hospitalError } = await supabase
        .from('hospitals')
        .update({
          name: hospitalData.name,
          address: hospitalData.address,
          contact_number: hospitalData.contactNumber
        })
        .eq('id', userProfile.hospital_id)

      if (hospitalError) throw hospitalError

      // Update email if changed
      if (userData.email !== user?.email) {
        const { error: emailError } = await supabase.auth.updateUser({
          email: userData.email
        })
        if (emailError) throw emailError
      }

      await refreshUserProfile()
      setSuccess('Settings updated successfully!')
      setTimeout(() => setSuccess(''), 3000)
    } catch (error: any) {
      setError(error.message || 'Failed to update settings')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-6 max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
        <p className="text-gray-600 mt-1">Manage your account and hospital information</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-8">
        {success && (
          <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded-lg text-sm">
            {success}
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm">
            {error}
          </div>
        )}

        {/* User Information */}
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <div className="flex items-center space-x-3 mb-6">
            <User className="w-6 h-6 text-teal-600" />
            <h2 className="text-xl font-semibold text-gray-900">User Information</h2>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Username
              </label>
              <input
                type="text"
                name="username"
                value={userData.username}
                onChange={handleUserDataChange}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-teal-500 transition-colors"
                placeholder="Your full name"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Email Address
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="email"
                  name="email"
                  value={userData.email}
                  onChange={handleUserDataChange}
                  className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-teal-500 transition-colors"
                  placeholder="your.email@hospital.com"
                  required
                />
              </div>
            </div>

            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Contact Information
              </label>
              <div className="relative">
                <Phone className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="text"
                  name="contactInfo"
                  value={userData.contactInfo}
                  onChange={handleUserDataChange}
                  className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-teal-500 transition-colors"
                  placeholder="Phone number or alternate contact"
                  required
                />
              </div>
            </div>
          </div>
        </div>

        {/* Hospital Information */}
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <div className="flex items-center space-x-3 mb-6">
            <Building className="w-6 h-6 text-teal-600" />
            <h2 className="text-xl font-semibold text-gray-900">Hospital Information</h2>
          </div>

          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Hospital Name
              </label>
              <input
                type="text"
                name="name"
                value={hospitalData.name}
                onChange={handleHospitalDataChange}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-teal-500 transition-colors"
                placeholder="General Hospital"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Hospital Address
              </label>
              <input
                type="text"
                name="address"
                value={hospitalData.address}
                onChange={handleHospitalDataChange}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-teal-500 transition-colors"
                placeholder="123 Medical Center Blvd, City, State"
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Hospital Contact Number
              </label>
              <div className="relative">
                <Phone className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="text"
                  name="contactNumber"
                  value={hospitalData.contactNumber}
                  onChange={handleHospitalDataChange}
                  className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-teal-500 transition-colors"
                  placeholder="(555) 123-4567"
                  required
                />
              </div>
            </div>
          </div>
        </div>

        {/* Security Section */}
        <div className="bg-white rounded-xl border border-gray-200 p-6">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">Security</h2>
          <p className="text-gray-600 mb-4">
            For security reasons, password changes must be done through the password reset flow.
          </p>
          <button
            type="button"
            onClick={() => {
              supabase.auth.resetPasswordForEmail(user?.email!, {
                redirectTo: `${window.location.origin}/reset-password`,
              })
              setSuccess('Password reset email sent!')
            }}
            className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
          >
            Reset Password
          </button>
        </div>

        {/* Submit Button */}
        <div className="flex justify-end">
          <button
            type="submit"
            disabled={loading}
            className="px-6 py-3 bg-teal-600 text-white rounded-lg hover:bg-teal-700 focus:ring-4 focus:ring-teal-200 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
          >
            {loading ? (
              <>
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                <span>Saving...</span>
              </>
            ) : (
              <>
                <Save className="w-5 h-5" />
                <span>Save Changes</span>
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  )
}

export default Settings