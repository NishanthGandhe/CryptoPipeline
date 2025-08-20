import { NextRequest, NextResponse } from 'next/server';
import { exec } from 'child_process';
import { promisify } from 'util';
import path from 'path';
import fs from 'fs';

const execAsync = promisify(exec);

export async function POST(request: NextRequest) {
  try {
    console.log('Starting data refresh pipeline...');
    
    // Path to the pipeline directory
    const pipelineDir = path.join(process.cwd(), '..', 'pipeline');
    
    // Check if pipeline directory exists
    if (!fs.existsSync(pipelineDir)) {
      throw new Error(`Pipeline directory not found: ${pipelineDir}`);
    }
    
    // Check if required scripts exist
    const ingestPath = path.join(pipelineDir, 'ingest.py');
    const insightPath = path.join(pipelineDir, 'generate_insight.py');
    
    if (!fs.existsSync(ingestPath)) {
      throw new Error(`ingest.py not found at: ${ingestPath}`);
    }
    
    if (!fs.existsSync(insightPath)) {
      throw new Error(`generate_insight.py not found at: ${insightPath}`);
    }
    
    console.log(`Pipeline directory: ${pipelineDir}`);
    console.log(`Running ingest.py from: ${ingestPath}`);
    
    // First run ingest.py to get fresh data
    const ingestResult = await execAsync('python3 ingest.py', {
      cwd: pipelineDir,
      timeout: 300000, // 5 minute timeout
      env: { ...process.env } // Pass environment variables
    });
    
    console.log('Ingest completed successfully');
    console.log('Ingest stdout:', ingestResult.stdout);
    if (ingestResult.stderr) {
      console.log('Ingest stderr:', ingestResult.stderr);
    }
    
    console.log(`Running generate_insight.py from: ${insightPath}`);
    
    // Then run generate_insight.py to update forecasts
    const insightResult = await execAsync('python3 generate_insight.py', {
      cwd: pipelineDir,
      timeout: 600000, // 10 minute timeout for ML processing
      env: { ...process.env } // Pass environment variables
    });
    
    console.log('Insight generation completed successfully');
    console.log('Insight stdout:', insightResult.stdout);
    if (insightResult.stderr) {
      console.log('Insight stderr:', insightResult.stderr);
    }
    
    return NextResponse.json({ 
      success: true, 
      message: 'Data pipeline completed successfully',
      timestamp: new Date().toISOString(),
      details: {
        pipelineDir,
        ingest: {
          stdout: ingestResult.stdout.slice(-1000), // Last 1000 chars to avoid huge responses
          stderr: ingestResult.stderr?.slice(-1000) || null
        },
        insight: {
          stdout: insightResult.stdout.slice(-1000),
          stderr: insightResult.stderr?.slice(-1000) || null
        }
      }
    });
    
  } catch (error: any) {
    console.error('Pipeline execution failed:', error);
    
    // More detailed error information
    const errorDetails = {
      message: error.message,
      code: error.code,
      killed: error.killed,
      signal: error.signal,
      cmd: error.cmd,
      stdout: error.stdout?.slice(-1000) || null,
      stderr: error.stderr?.slice(-1000) || null
    };
    
    return NextResponse.json({ 
      success: false, 
      message: 'Data pipeline failed',
      timestamp: new Date().toISOString(),
      error: errorDetails
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
