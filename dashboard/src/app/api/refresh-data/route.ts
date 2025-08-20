import { NextRequest, NextResponse } from 'next/server';

// Get backend URL from environment variable
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || process.env.BACKEND_URL || 'http://localhost:8000';

// eslint-disable-next-line @typescript-eslint/no-unused-vars
export async function POST(_request: NextRequest) {
  try {
    console.log('Triggering data refresh via backend API...');
    console.log('Backend URL:', BACKEND_URL);
    
    // Call the backend's refresh endpoint (synchronous version that waits for completion)
    const response = await fetch(`${BACKEND_URL}/refresh-data-sync`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      // 15 minute timeout for the full pipeline
      signal: AbortSignal.timeout(900000)
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Backend API error: ${response.status} ${response.statusText} - ${errorText}`);
    }
    
    const result = await response.json();
    
    console.log('Backend refresh completed successfully');
    
    return NextResponse.json({ 
      success: true, 
      message: 'Data pipeline completed successfully via backend',
      timestamp: new Date().toISOString(),
      backend_response: result
    });
    
  } catch (error: unknown) {
    console.error('Backend API call failed:', error);
    
    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    
    // Handle timeout specifically
    if (error instanceof Error && error.name === 'TimeoutError') {
      return NextResponse.json({ 
        success: false, 
        message: 'Data pipeline timed out (still may be running in background)',
        timestamp: new Date().toISOString(),
        error: {
          type: 'timeout',
          message: errorMessage,
          suggestion: 'Try refreshing the page in a few minutes to see if data was updated'
        }
      }, { status: 504 });
    }
    
    // Handle network errors
    if (error instanceof Error && (error.message.includes('fetch') || error.message.includes('ECONNREFUSED'))) {
      return NextResponse.json({ 
        success: false, 
        message: 'Could not connect to backend service',
        timestamp: new Date().toISOString(),
        error: {
          type: 'connection',
          message: errorMessage,
          backend_url: BACKEND_URL,
          suggestion: 'Backend service may be starting up or unavailable'
        }
      }, { status: 503 });
    }
    
    return NextResponse.json({ 
      success: false, 
      message: 'Data pipeline failed',
      timestamp: new Date().toISOString(),
      error: {
        type: 'unknown',
        message: errorMessage
      }
    }, { status: 500 });
  }
}

// Add a GET endpoint to check pipeline status
export async function GET() {
  return NextResponse.json({ 
    message: 'Data refresh endpoint is ready',
    endpoints: {
      refresh: 'POST /api/refresh-data - Triggers full data pipeline',
      status: 'GET /api/refresh-data - Check endpoint status'
    }
  });
}
