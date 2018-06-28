# Flask Vs Tornado


I always thought using Flask & Tornado together was stupid, but it actually does make sense. It adds complexity though; my preference would be to just use Tornado, but if you're attached to Flask, then this setup works.

Flask is (reportedly) very nice to use, and simpler than Tornado. However, Flask requires a WSGI server for production (or FCGI, but that's more complicated). Tornado is pretty simple to setup as a WSGI server: 


## what is  WSGI (Web Server Gateway Interface)?
https://en.wikipedia.org/wiki/Web_Server_Gateway_Interface

https://www.fullstackpython.com/wsgi-servers.html



http://docs.python-guide.org/en/latest/scenarios/web/


http://klen.github.io/py-frameworks-bench/

http://flask.pocoo.org/docs/0.10/deploying/wsgi-standalone/#tornado