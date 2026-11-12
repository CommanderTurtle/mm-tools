#!/usr/bin/env node

import crypto from 'node:crypto'
import http from 'node:http'
import https from 'node:https'

const backend = new URL(
  process.env.NOMAS_HTTP_PROXY_BACKEND ?? 'http://127.0.0.1:18084',
)
const host = process.env.NOMAS_HTTP_PROXY_HOST ?? '0.0.0.0'
const port = Number.parseInt(process.env.NOMAS_HTTP_PROXY_PORT ?? '8081', 10)
const username = process.env.NOMAS_DEMO_USERNAME ?? 'hy'
const password = process.env.NOMAS_DEMO_PASSWORD ?? ''

if (!password) {
  throw new Error('NOMAS_DEMO_PASSWORD is required')
}

const expectedAuthorization = Buffer.from(
  `Basic ${Buffer.from(`${username}:${password}`).toString('base64')}`,
)
const publicPrefixes = ['/share/', '/public/', '/fonts/']
const publicPaths = new Set(['/hunyuan-logo.png'])
const hopByHop = new Set([
  'connection',
  'keep-alive',
  'proxy-authenticate',
  'proxy-authorization',
  'te',
  'trailers',
  'transfer-encoding',
  'upgrade',
])

function isPublicRequest(request) {
  if (!['GET', 'HEAD'].includes(request.method ?? '')) return false
  const path = new URL(request.url ?? '/', 'http://localhost').pathname
  return publicPaths.has(path) || publicPrefixes.some((prefix) => path.startsWith(prefix))
}

function isAuthorized(request) {
  const supplied = Buffer.from(request.headers.authorization ?? '')
  return supplied.length === expectedAuthorization.length
    && crypto.timingSafeEqual(supplied, expectedAuthorization)
}

const server = http.createServer((request, response) => {
  if (!isPublicRequest(request) && !isAuthorized(request)) {
    response.writeHead(401, {
      'Cache-Control': 'no-store',
      'Content-Type': 'text/plain; charset=utf-8',
      'WWW-Authenticate': 'Basic realm="Hy-Nomas Demo", charset="UTF-8"',
    })
    response.end('Authentication required\n')
    return
  }

  const target = new URL(request.url ?? '/', backend)
  const headers = {}
  for (const [key, value] of Object.entries(request.headers)) {
    if (
      value !== undefined
      && !hopByHop.has(key.toLowerCase())
      && !['authorization', 'host'].includes(key.toLowerCase())
    ) {
      headers[key] = value
    }
  }
  const peer = request.socket.remoteAddress ?? ''
  const prior = request.headers['x-forwarded-for']
  headers['x-forwarded-for'] = prior ? `${prior}, ${peer}` : peer
  headers['x-forwarded-proto'] = 'http'
  headers['x-forwarded-host'] = request.headers.host ?? ''

  const transport = target.protocol === 'https:' ? https : http
  const upstream = transport.request(
    target,
    {
      method: request.method,
      headers,
    },
    (upstreamResponse) => {
      const responseHeaders = {}
      for (const [key, value] of Object.entries(upstreamResponse.headers)) {
        if (value !== undefined && !hopByHop.has(key.toLowerCase())) {
          responseHeaders[key] = value
        }
      }
      responseHeaders['x-content-type-options'] = 'nosniff'
      responseHeaders['referrer-policy'] = 'same-origin'
      response.writeHead(upstreamResponse.statusCode ?? 502, responseHeaders)
      upstreamResponse.pipe(response)
    },
  )

  upstream.on('error', (error) => {
    if (!response.headersSent) {
      response.writeHead(502, { 'Content-Type': 'text/plain; charset=utf-8' })
    }
    response.end(`Upstream unavailable: ${error.name}\n`)
  })
  request.pipe(upstream)
})

server.requestTimeout = 0
server.headersTimeout = 65_000
server.listen(port, host, () => {
  console.log(`NOMAS auth proxy listening on http://${host}:${port}`)
  console.log(`Forwarding to ${backend.origin}`)
})
