import React, { useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import { supabase } from '../lib/supabase'
import { 
  Brain, 
  Thermometer, 
  Zap, 
  Activity, 
  Gauge, 
  Droplets, 
  Clock, 
  AlertTriangle, 
  CheckCircle, 
  ArrowLeft,
  Save,
  BarChart3,
  Magnet,
  Wind,
  Cpu
} from 'lucide-react'

interface MRISensorReading {
  helium_level: number
  magnetic_field: number
  rf_power: number
  gradient_temp: number
  compressor_pressure: number
  chiller_flow: number
  room_humidity: number
  system_runtime: number
  vibration: number
  coil_temperature: number
  power_consumption: number
  magnet_quench_risk: number
}

interface MRIPrediction {
  condition: string
  failure: string
  cause: string
  solution: string
  time_to_failure: number
  confidence: number
  probabilities: string
}

const MRIMaintenanceLog: React.FC = () => {
  const { deviceId } = useParams<{ deviceId: string }>()
  const navigate = useNavigate()
  const { user } = useAuth()
  
  const [sensorReading, setSensorReading] = useState<MRISensorReading>({
    helium_level: 89.5,
    magnetic_field: 3.0,
    rf_power: 30.0,
    gradient_temp: 22.0,
    compressor_pressure: 15.0,
    chiller_flow: 55.0,
    room_humidity: 50.0,
    system_runtime: 2000.0,
    vibration: 1.0,
    coil_temperature: 25.0,
    power_consumption: 50.0,
    magnet_quench_risk: 0.05
  })
  
  const [prediction, setPrediction] = useState<MRIPrediction | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isSaving, setIsSaving] = useState(false)

  const handleInputChange = (field: keyof MRISensorReading, value: number) => {
    setSensorReading(prev => ({
      ...prev,
      [field]: value
    }))
  }

  const makeMRIPrediction = async (): Promise<MRIPrediction> => {
    try {
      // Call the Python backend API
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          device_type: 'MRI Scanner',
          helium_level: sensorReading.helium_level,
          magnetic_field: sensorReading.magnetic_field,
          rf_power: sensorReading.rf_power,
          gradient_temp: sensorReading.gradient_temp,
          compressor_pressure: sensorReading.compressor_pressure,
          chiller_flow: sensorReading.chiller_flow,
          room_humidity: sensorReading.room_humidity,
          system_runtime: sensorReading.system_runtime,
          vibration: sensorReading.vibration,
          coil_temperature: sensorReading.coil_temperature,
          power_consumption: sensorReading.power_consumption,
          magnet_quench_risk: sensorReading.magnet_quench_risk
        })
      })

      if (!response.ok) {
        throw new Error(`API Error: ${response.status}`)
      }

      const result = await response.json()
      
      if (result.error) {
        throw new Error(result.error)
      }

      // Map API response condition to database expected values
      const conditionMapping: { [key: string]: string } = {
        'Normal': 'Good',
        'Good': 'Good',
        'Average': 'Average', 
        'Warning': 'Warning',
        'Critical': 'Critical'
      }

      return {
        ...result,
        condition: conditionMapping[result.condition] || 'Good'
      }
    } catch (error: any) {
      console.error('Prediction error:', error)
      // Fallback to simulation if API is not available
      const mockPrediction = simulateMRIPrediction()
      throw new Error('API unavailable, using simulation mode. ' + error.message)
    }
  }

  const simulateMRIPrediction = (): MRIPrediction => {
    // Simple rule-based simulation (fallback)
    const avgValue = Object.values(sensorReading).reduce((a, b) => a + b, 0) / Object.values(sensorReading).length
    const condition = avgValue > 50 ? 'Good' : avgValue > 30 ? 'Warning' : 'Critical'
    const failure = condition === 'Critical' ? 'Yes' : 'No'
    const confidence = Math.random() * 40 + 60 // 60-100%
    
    return {
      condition,
      failure,
      cause: condition === 'Critical' ? 'Cryogenic System Failure' : 'Normal Operation',
      solution: condition === 'Critical' ? 'Check helium supply system' : 'Continue monitoring',
      time_to_failure: Math.random() * 200 + 100,
      confidence,
      probabilities: `Good: ${Math.random() * 30 + 40}%\nAverage: ${Math.random() * 20 + 20}%\nWarning: ${Math.random() * 10 + 10}%\nCritical: ${Math.random() * 5 + 5}%`
    }
  }

  const handleAnalyze = async () => {
    setIsAnalyzing(true)
    try {
      const result = await makeMRIPrediction()
      setPrediction(result)
    } catch (error: any) {
      console.error('Analysis failed:', error)
      // Fallback to simulation
      const mockResult = simulateMRIPrediction()
      setPrediction(mockResult)
      alert('API unavailable, using simulation mode. ' + error.message)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleSubmit = async () => {
    if (!prediction || !deviceId || !user) return
    
    setIsSaving(true)
    try {
      // First, create maintenance log entry
      const { data: maintenanceLog, error: logError } = await supabase
        .from('maintenance_logs')
        .insert({
          device_id: deviceId,
          uploaded_by: user.id,
          date_of_maintenance: new Date().toISOString(),
          notes: `MRI Maintenance Log - AI Analysis: ${prediction.condition} condition, TTF: ${prediction.time_to_failure} days`
        })
        .select()
        .single()

      if (logError) throw logError

      const maintenanceLogId = maintenanceLog.id

      // Save prediction data to mri_predictions table
      const { data: predictionResult, error: predictionError } = await supabase
        .from('mri_predictions')
        .insert({
          device_id: deviceId,
          maintenance_log_id: maintenanceLogId,
          uploaded_by: user.id,
          helium_level: sensorReading.helium_level,
          magnetic_field: sensorReading.magnetic_field,
          rf_power: sensorReading.rf_power,
          gradient_temp: sensorReading.gradient_temp,
          compressor_pressure: sensorReading.compressor_pressure,
          chiller_flow: sensorReading.chiller_flow,
          room_humidity: sensorReading.room_humidity,
          system_runtime: sensorReading.system_runtime,
          vibration: sensorReading.vibration,
          coil_temperature: sensorReading.coil_temperature,
          power_consumption: sensorReading.power_consumption,
          magnet_quench_risk: sensorReading.magnet_quench_risk,
          predicted_condition: prediction.condition,
          predicted_failure: prediction.failure,
          predicted_cause: prediction.cause,
          recommended_solution: prediction.solution,
          time_to_failure_days: prediction.time_to_failure,
          confidence_score: prediction.confidence,
          class_probabilities: parseProbabilitiesToJSON(prediction.probabilities)
        })
        .select()

      if (predictionError) {
        console.error('Error saving prediction:', predictionError)
        console.error('Prediction data that failed:', {
          device_id: deviceId,
          maintenance_log_id: maintenanceLogId,
          uploaded_by: user.id,
          // ... other fields
        })
        throw new Error(`Failed to save prediction: ${predictionError.message}`)
      }

      console.log('MRI prediction saved successfully:', predictionResult)

      // Update device status based on AI prediction
      const deviceStatus = getDeviceStatusFromPrediction(prediction.condition)
      const nextMaintenanceDate = calculateNextMaintenanceDate(prediction.time_to_failure)

      const { error: updateError } = await supabase
        .from('devices')
        .update({
          status: deviceStatus,
          last_maintenance_date: new Date().toISOString(),
          next_maintenance_date: nextMaintenanceDate.toISOString()
        })
        .eq('id', deviceId)

      if (updateError) {
        console.error('Error updating device status:', updateError)
      }

      navigate('/dashboard')
      
    } catch (error: any) {
      console.error('Error saving MRI maintenance log:', error)
      
      let errorMessage = 'Failed to save MRI maintenance log. Please try again.'
      
      if (error.message?.includes('relation "mri_predictions" does not exist')) {
        errorMessage = 'MRI predictions table not found. Please ensure the database migration has been applied.'
      } else if (error.message?.includes('permission denied')) {
        errorMessage = 'Permission denied. Please check your user permissions.'
      } else if (error.message?.includes('foreign key')) {
        errorMessage = 'Invalid device or maintenance log reference. Please try again.'
      }
      
      alert(errorMessage)
    } finally {
      setIsSaving(false)
    }
  }

  const getDeviceStatusFromPrediction = (condition: string): 'OK' | 'Warning' | 'Danger' => {
    switch (condition) {
      case 'Good': return 'OK'
      case 'Normal': return 'OK'
      case 'Average': return 'OK'
      case 'Warning': return 'Warning'
      case 'Critical': return 'Danger'
      default: return 'OK'
    }
  }

  const calculateNextMaintenanceDate = (timeToFailure: number): Date => {
    const nextMaintenance = new Date()
    nextMaintenance.setDate(nextMaintenance.getDate() + Math.min(timeToFailure, 90)) // Cap at 90 days
    return nextMaintenance
  }

  const parseProbabilitiesToJSON = (probabilitiesString: string) => {
    try {
      const lines = probabilitiesString.trim().split('\n')
      const probabilities: { [key: string]: number } = {}
      
      for (const line of lines) {
        if (line.includes(':')) {
          const [className, percentage] = line.split(':')
          const cleanClassName = className.trim()
          const cleanPercentage = parseFloat(percentage.trim().replace('%', ''))
          probabilities[cleanClassName] = cleanPercentage
        }
      }
      
      return probabilities
    } catch (error) {
      console.error('Error parsing probabilities:', error)
      return { Good: 0, Average: 0, Warning: 0, Critical: 0 }
    }
  }

  const sensorFields = [
    { key: 'helium_level', label: 'Helium Level', unit: '%', icon: Wind, range: [60, 95] },
    { key: 'magnetic_field', label: 'Magnetic Field', unit: 'T', icon: Magnet, range: [2.5, 3.1] },
    { key: 'rf_power', label: 'RF Power', unit: 'kW', icon: Zap, range: [20, 60] },
    { key: 'gradient_temp', label: 'Gradient Temperature', unit: '°C', icon: Thermometer, range: [15, 55] },
    { key: 'compressor_pressure', label: 'Compressor Pressure', unit: 'bar', icon: Gauge, range: [3, 20] },
    { key: 'chiller_flow', label: 'Chiller Flow', unit: 'L/min', icon: Droplets, range: [15, 70] },
    { key: 'room_humidity', label: 'Room Humidity', unit: '%', icon: Droplets, range: [25, 90] },
    { key: 'system_runtime', label: 'System Runtime', unit: 'hours', icon: Clock, range: [0, 15000] },
    { key: 'vibration', label: 'Vibration', unit: 'mm/s', icon: Activity, range: [0.1, 8.0] },
    { key: 'coil_temperature', label: 'Coil Temperature', unit: '°C', icon: Thermometer, range: [18, 55] },
    { key: 'power_consumption', label: 'Power Consumption', unit: 'kW', icon: Cpu, range: [40, 95] },
    { key: 'magnet_quench_risk', label: 'Magnet Quench Risk', unit: '', icon: AlertTriangle, range: [0.01, 0.6] }
  ]

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <button
            onClick={() => navigate('/dashboard')}
            className="flex items-center text-gray-600 hover:text-gray-900 mb-4"
          >
            <ArrowLeft className="w-5 h-5 mr-2" />
            Back to Dashboard
          </button>
          
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 flex items-center">
                <Brain className="w-8 h-8 mr-3 text-blue-600" />
                MRI Scanner Maintenance Log
              </h1>
              <p className="text-gray-600 mt-2">AI-powered failure prediction and maintenance tracking</p>
            </div>
            
            <div className="flex space-x-3">
              <button
                onClick={handleAnalyze}
                disabled={isAnalyzing}
                className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
              >
                <BarChart3 className="w-5 h-5 mr-2" />
                {isAnalyzing ? 'Analyzing...' : 'Analyze with AI'}
              </button>
              
              {prediction && (
                <button
                  onClick={handleSubmit}
                  disabled={isSaving}
                  className="flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50"
                >
                  <Save className="w-5 h-5 mr-2" />
                  {isSaving ? 'Saving...' : 'Save Log'}
                </button>
              )}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Sensor Input */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-6">MRI Sensor Readings</h2>
            
            <div className="space-y-4">
              {sensorFields.map((field) => {
                const Icon = field.icon
                return (
                  <div key={field.key} className="flex items-center space-x-4">
                    <Icon className="w-5 h-5 text-gray-400 flex-shrink-0" />
                    <div className="flex-1">
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        {field.label}
                      </label>
                      <div className="flex items-center space-x-2">
                        <input
                          type="number"
                          value={sensorReading[field.key as keyof MRISensorReading]}
                          onChange={(e) => handleInputChange(field.key as keyof MRISensorReading, parseFloat(e.target.value))}
                          min={field.range[0]}
                          max={field.range[1]}
                          step="0.1"
                          className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                        />
                        <span className="text-sm text-gray-500 w-12">{field.unit}</span>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          {/* AI Prediction Results */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-6">AI Analysis Results</h2>
            
            {prediction ? (
              <div className="space-y-6">
                {/* Condition Status */}
                <div className="text-center">
                  <div className={`inline-flex items-center px-4 py-2 rounded-full text-lg font-semibold ${
                    prediction.condition === 'Good' || prediction.condition === 'Normal' ? 'bg-green-100 text-green-800' :
                    prediction.condition === 'Average' ? 'bg-blue-100 text-blue-800' :
                    prediction.condition === 'Warning' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    {prediction.condition === 'Good' || prediction.condition === 'Normal' ? <CheckCircle className="w-5 h-5 mr-2" /> :
                     prediction.condition === 'Average' ? <CheckCircle className="w-5 h-5 mr-2" /> :
                     prediction.condition === 'Warning' ? <AlertTriangle className="w-5 h-5 mr-2" /> :
                     <AlertTriangle className="w-5 h-5 mr-2" />}
                    {prediction.condition} Condition
                  </div>
                </div>

                {/* Prediction Details */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="text-sm text-gray-600">Failure Risk</div>
                    <div className={`text-lg font-semibold ${
                      prediction.failure === 'No' ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {prediction.failure}
                    </div>
                  </div>
                  
                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="text-sm text-gray-600">Confidence</div>
                    <div className="text-lg font-semibold text-blue-600">
                      {prediction.confidence.toFixed(1)}%
                    </div>
                  </div>
                  
                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="text-sm text-gray-600">Time to Failure</div>
                    <div className="text-lg font-semibold text-orange-600">
                      {prediction.time_to_failure.toFixed(0)} days
                    </div>
                  </div>
                  
                  <div className="bg-gray-50 rounded-lg p-4">
                    <div className="text-sm text-gray-600">Root Cause</div>
                    <div className="text-sm font-medium text-gray-900">
                      {prediction.cause}
                    </div>
                  </div>
                </div>

                {/* Recommended Solution */}
                <div>
                  <div className="text-sm font-medium text-gray-700 mb-2">Recommended Solution</div>
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                    <p className="text-sm text-blue-800">{prediction.solution}</p>
                  </div>
                </div>

                {/* Probabilities */}
                <div>
                  <div className="text-sm font-medium text-gray-700 mb-2">Condition Probabilities</div>
                  <div className="bg-gray-50 rounded-lg p-3">
                    <pre className="text-sm text-gray-700 whitespace-pre-wrap">{prediction.probabilities}</pre>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <Brain className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                <p className="text-gray-500">Click "Analyze with AI" to get MRI failure prediction</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default MRIMaintenanceLog

