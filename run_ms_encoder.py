########### Python 3.2 #############
import http.client, urllib.request, urllib.parse, urllib.error, base64

headers = {
    # Request headers
    'Content-Type': 'application/json',
    'Ocp-Apim-Subscription-Key': '924c1505854b4da4a6144a1cce92937f',
}

params = urllib.parse.urlencode({
})

try:
    conn = http.client.HTTPSConnection('api.msturing.org')
    conn.request("POST", "/gen/encode?%s" % params, '{"queries": ["Microsoft", "Bing.com", "Azure Cloud Services"]}', headers)
    response = conn.getresponse()
    data = response.read()
    print(data)
    conn.close()
except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))