import http.client, urllib.request, urllib.parse, urllib.error, base64

headers = {
    # Request headers
    'Content-Type': 'application/json',
    'Ocp-Apim-Subscription-Key': '3d886bf041574645a097aab80bc26f0e',
}

params = urllib.parse.urlencode({})
conn = http.client.HTTPSConnection('api.msturing.org')
conn.request("POST", "/gen/encode?%s" % params, '{"queries": ["Microsoft", "Bing.com", "Azure Cloud Services"]}', headers)
response = conn.getresponse()
data = response.read()
print(data)
print(response.status, response.reason)
conn.close()
