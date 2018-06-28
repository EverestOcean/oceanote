---
layout: post
title:  "在公有云上搭建负载均衡的Docker私有仓库"
date:   2015-09-28 09:00:00 +0800
categories: [docker,cloud-computing,operation]
---

<h2>简介</h2>

随着Docker的普及和在不同企业的深入应用，对于如何更安全，更快速的管理及存储 丰富多样的Docker私有镜像成为使用者亟需解决的问题。目前对于这个问题，Docker官方提供了公共的Docker Hub为用户管理Docker 镜像。国内也有京东开源的Speedy， DaoCloud提供的Docker Hub等。

除此以外的另一种常用方案是搭建自己的私有镜像仓库。本文以阿里云的环境为例，将介绍如何在Ubuntu12.04上基于Docker Registry V2，阿里云OSS， Nginx [4]搭建负载均衡自己的Docker镜像私有仓库, 如图下图.

![docker_registry](/assets/images/docker_registry/docker_registry.png)


<h2>目录</h2>

* [1. Docker Registry Server端配置](#1)
* [2. Docker Registry Client端配置](#2)
* [3. 私有Docker Registry使用方法](#3)
* [4. 参考文献](#4)
* [5. 附录](#5)

<h3 id="1">1. Docker Registry Server端配置</h3>

<h4>1.1 安装依赖</h4>

安装apache2-utils 包 (该包可以用来更简单的创建nginx身份验证文件，见 htpasswd)

```
sudo apt-get update
sudo apt-get install -y apache2-utils
```

<h4>1.2 配置SSL</h4>

这边使用SSL 证书是自生成的临时证书，如果需要更高的安全保障，建议注册官方证书。 同时需要注意在使用自生成证书，必须在所有的docker daemon中配置(包括服务端和客户端)。

* 创建本地文件夹，用于存储临时证书

	```
	mkdir ~/certs
	cd ~/certs
	```
* 生成新额根密钥

	```
	sudo openssl genrsa -out dockerCA.key 2048
	```
* 生成新的根证书

	```
	sudo openssl req -x509 -new -nodes -key dockerCA.key -days 10000 -out dockerCA.crt
	```

	在交互配置中输入相应的信息，其中 Common Name 需要输入docker registry 服务器的域名，例如： docker.xx.com [3]

	![docker_register_1](/assets/images/docker_registry/docker_registry_1.png)

* 为nginx web服务生成SSL密钥

	```
	sudo openssl genrsa -out nginx.key 2048
	```

* 为nginx 生成证书

	```
	sudo openssl req -new -key nginx.key -out nginx.csr
	```

	**注**：在交互配置中输入相应的信息，其中 Common Name 需要输入docker registry 服务器的域名，例如： docker.xx.com, 与根密钥相同， 不需要输入 challenge password

	![docker_register_2](/assets/images/docker_registry/docker_registry_2.png)

* 私有CA 根据请求来签发证书

	```
	sudo openssl x509 -req -in nginx.csr -CA dockerCA.crt -CAkey dockerCA.key -CAcreateserial -out nginx.crt -days 10000
	```

<h4>1.3 安装，配置 nginx 1.9.4 </h4>

* 添加组和用户

	```
	sudo groupadd www -g 58
	sudo useradd -u 58 -g www www
	```

* 下载nginx 源文件

	```
	wget http://nginx.org/download/nginx-1.9.4.tar.gz
	```
	
* 编译，安装 nginx

	```
	tar zxvf ./nginx-1.9.4.tar.gz
	cd ./nginx-1.9.4
	./configure --user=www --group=www --prefix=/opt/nginx  --with-pcre --with-http_stub_status_module --with-http_ssl_module --with-http_addition_module --with-http_realip_module --with-http_flv_module
	sudo make
	sudo make install
	cd `pwd`
	rm -rf `pwd`/nginx-1.9.4/
	rm -rf `pwd`/nginx-1.9.4.tar.gz
	```

* 生成 htpasswd

	```
	sudo htpasswd -cb /opt/nginx/conf/.htpasswd ${USER} ${PASSWORD}
	```

	假设 \$\{USER\}=”test”  \$\{PASSWORD\}=”1234”, 后面docker 客户端登陆时需要使用的用户名和密码，可以使用下面的命令增加用户名和密码

	```
	sudo htpasswd -b /opt/nginx/conf/.htpasswd ${USER_2} ${PASSWORD_2}
	```

	可以使用下面命令删除已有用户

	```
	htpasswd -D /opt/nginx/conf/.htpasswd ${USER}
	```

* 将生成的 nginx SSL密钥和证书放置入nginx.conf中指定的目录

	在本例中目录为 /etc/nginx/ssl/

	```
	sudo cp ~/certs/nginx.key /etc/nginx/ssl/
	sudo cp ~/certs/nginx.crt /etc/nginx/ssl/
	```

* 编辑/opt/nginx/conf/nginx.conf 文件

	```
	# daemon off;
	# 使用的用户和组
	user  www www;    
	# 指定工作进程数(一般等于CPU总核数)
	worker_processes  auto;
	# 指定错误日志的存放路径,错误日志记录级别选项为:[debug | info | notic | warn | error | crit]
	error_log  /var/log/nginx_error.log  error;
	# 指定pid存放的路径
	# pid        logs/nginx.pid;
	# 指定文件描述符数量
	worker_rlimit_nofile 51200;
	events {
   		# 使用的网络I/O模型,Linux推荐epoll;FreeBSD推荐kqueue
   		use epoll;
   		# 允许的最大连接数
   		worker_connections  51200;
   		multi_accept on;
	}  
	http {
   		include       mime.types;
   		log_format  main  '$remote_addr - $remote_user 		[$time_local] "$request" ' '$status $body_bytes_sent "$http_referer" ' '"$http_user_agent" 		"$upstream_addr"';
   		access_log  /var/log/nginx_access.log  main;
   		# 服务器名称哈希表的桶大小,该默认值取决于CPU缓存
   		server_names_hash_bucket_size 128;
   		# 客户端请求的Header头缓冲区大小
   		client_header_buffer_size 32k;
   		large_client_header_buffers 4 32k;
   		# 启用sendfile()函数
   		sendfile        on;
   		tcp_nopush      on;
   		tcp_nodelay     on;
   		keepalive_timeout  65;
   		upstream registry {
      		server 127.0.0.1:5000;
   		}
   		server {
      		listen       443;
      		server_name  192.168.1.100;
      		ssl                  on;
      		ssl_certificate /etc/nginx/ssl/nginx.crt;
      		ssl_certificate_key /etc/nginx/ssl/nginx.key;
      		client_max_body_size 0; # disable any limits to avoid HTTP 413 for large image uploads
      		# required to avoid HTTP 411: see Issue #1486 	(https://github.com/docker/docker/issues/1486)
      		chunked_transfer_encoding on;
      		location /v2/ {
          		if ($http_user_agent ~ "^(docker\/1\.(3|4|5(?!\.[0-9]-dev))|Go ).*\$" ) {
              		return 404;
          		}
          auth_basic "registry";
          auth_basic_user_file /opt/nginx/conf/.htpasswd;
          add_header 'Docker-Distribution-Api-Version' 'registry/2.0' always;
          proxy_pass       http://registry;
          proxy_set_header  Host              $http_host;
          proxy_set_header  X-Real-IP         $remote_addr;
          proxy_set_header  X-Forwarded-For   $proxy_add_x_forwarded_for;
          proxy_set_header  X-Forwarded-Proto $scheme;
          proxy_read_timeout                  900;
      		}
      		location /_ping {
          		auth_basic off;
          		proxy_pass http://registry;
      		}
      		location /v1/_ping {
          		auth_basic off;
          		proxy_pass http://registry;
       		}
   		}
	}
	```

	**注：** 该配置访问的是localhost私有仓库，并没有加入负载均衡

* nginx 负载均衡配置

	假设有两台服务器：

	```
	registry_1: 192.168.1.101    
	registry_2: 192.168.1.102
	```	
	
	则 编辑 /opt/nginx/conf/nginx.conf 文件

	```
	upstream registry {
    	ip_hash;
    	server 192.168.1.101:5000 weight=2;
    	server 192.168.1.102:5000 down;
  	}
	```

	**注：** 负载均衡支持的分配方式

		1. 轮询（默认）: 每个请求按时间顺序逐一分配到不同的后端服务器上，如果后端服务down掉，能自动删除
		2. weight: 指定轮询几率，weight和访问比率成正比，用于后端服务器性能不均匀的情况
		3. ip_hash: 每个请求按访问ip的hash结果分配，这样每个访客固定访问一个后端服务，可以解决session的问题
		4. fair: 安装服务器的响应时间来分配请求，响应时间短的优先分配
		5. url_hash: 每个请求按照访问url 的hash结果分配
		6. down: 该服务器不参与被访问

* 验证nginx 配置是否正确

	```
	sudo /opt/nginx/sbin/nginx -t
	```

* 启动 nginx

	```
	sudo /opt/nginx/sbin/nginx
	```

* 验证nginx是否启动

	```
	ps -ef|grep -i "nginx"
	```

<h4>1.4 配置，运行 Docker </h4>

* 停止 docker
	
	```
	sudo service docker stop
	```

* 编辑 /etc/default/docker 文件，加入如下一行

	```
	DOCKER_OPTS="--insecure-registry docker.xx.com"
	```

* 将自生成根证书变成合法证书

	由于使用的是自生产证书，所以我们需要告诉所有的客户端，该证书是合法的证书，并且同时告诉docker registry server该证书是合法的。

	```
	sudo mkdir -p /usr/local/share/ca-certificates/docker-dev-cert
	sudo cp ~/certs/dockerCA.crt /usr/local/share/ca- 	certificates/docker-dev-cert
	sudo update-ca-certificates
	```

	验证: 查看 /ect/ssl/certs/dockerCA.pem 是否存在，存在则成功。

* 将自生成的根证书加入 /etc/docker/certs.d/docker.xx.com/ca.crt 或者 ~/.docker/certs.d/docker.xx.com/ca.crt

	```
	sudo mkdir -p /etc/docker/certs.d/docker.xx.com/
	sudo cp dockerCA.crt /etc/docker/certs.d/docker.xx.com/ca.crt
	```

* 启动 docker

	```
	sudo service docker start
	```

<h4>1.5 配置hosts</h4>

* 编辑 /etc/hosts, 把docker.xx.com添加进 hosts

	```
	192.169.1.100 docker.xx.com
	```

<h4>1.6 编译，配置，运行 docker registry服务</h4>

* 编译 docker registry服务

	由于docker registry最近才支持 OSS的存储，所以最新官方的registry docker镜像中还没有包含OSS的相关代码，所以需要从github 拉取docker registry的代码，重新生成最新的image。如果后期更新则这步不需要做。


	```
	git clone git@github.com:docker/distribution.git
	cd ./distribution
	docker build --rm -t registry:latest .
	```

* 配置registry

本地创建 config.yml 文件，在里面加入

```
	
	version: 0.1
	log:
  		level: debug
  		formatter: text
  		fields:
    		service: registry
    		environment: staging
	storage:
  		oss:
    		accesskeyid: <your oss access id>
    		accesskeysecret: <your oss access key>
    		region: oss-cn-beijing #由您oss仓库隶属的区域决定，这边以北京为例
    		bucket: <your oss bucket>
    		rootdirectory: <root diectory>(optional) #用于存储的根路径，默认为空，则存储在 oss bucket下
  	delete:
    		enabled: false
  	redirect:
    		disable: false
  	cache:
    		blobdescriptor: inmemory #缓存方式，有两种选择，一种 in memory，另一种 redis
  	maintenance:
    		uploadpurging:
      			enabled: true
      			age: 168h
      			interval: 24h
      			dryrun: false
	http:
  			addr: 0.0.0.0:5000
  	debug:
    		addr: 0.0.0.0:5001
  	headers:
    		X-Content-Type-Options: [nosniff]
	
```

**注** oss详细的配置信息可以参考github, 同时 docker registry更详细的配置信息可以参考 Docker Registry Configuration

* 启动registry
 
	```
	docker run -d -p 127.0.0.1:5000:5000 --restart=always --name registry -v `pwd`/config.yml:/etc/docker/registry/config.yml  registry:latest
	```

* 验证registry

	```
	curl -i -k https://test:123@docker.xx.com
	```


<h3 id="2">2. Docker Registry Client端配置</h3>

* 编辑 /etc/hosts， 把 docker.xx.com的ip地址添加进去

	```
	sudo vim /etc/hosts
	192.168.1.100 docker.xx.com
	```

* 将在服务端创建的SSL 证书加入到客户端

	在服务端执行：
	```
	cat ~/certs/dockerCA.crt
	```

	在客户端，创建下面的文件夹，并更新CA证书：

	```
	sudo mkdir -p /usr/local/share/ca-certificates/docker-dev-cert/
	sudo vim  /usr/local/share/ca-certificates/docker-dev-cert/dockerCA.crt
	sudo update-ca-certificates
	```

* 停止 docker

	```
	sudo service docker stop
	```

* 编辑 /etc/default/docker 文件，加入下面一行

	```
	DOCKER_OPTS="--insecure-registry docker.xx.com"
	```

* 将根证书加入到 /etc/docker/certs.d/docker.xx.com/ca.crt 或者 ~/.docker/certs.d/docker.xx.com/ca.crt

	```
	sudo mkdir -p /etc/docker/certs.d/docker.xx.com/
	sudo cp dockerCA.crt /etc/docker/certs.d/docker.xx.com/ca.crt
	```

* 重启 docker

	```
	sudo service docker start
	```

<h3 id="3">3. 使用私有registry 步骤</h3>

* 登入： docker login -u test -p 123 -e “test@gmail.com“ https://docker.xx.com
* tag image名称： docker tag ubuntu:12.04 docker.xx.com/ubuntu:12.04
* 发布image： docker push docker.xx.com/ubuntu:12.04
* 下拉image： docker pull docker.xx.com/ubuntu:12.04


到此为止我们就搭建了一套私有docker镜像仓库。除了一般功能，还通过OSS和nignx服务实现了下载流量的负载均衡。如果你有一个比较大的计算集群依赖于docker服务，各计算节点同时下载更新镜像时，仓库服务不会成为瓶颈。同时使用自己搭建的私有镜像仓库，不仅能保证私有镜像的安全，而且pull 和push 镜像速度上比公共的Docker Hub快。


<h3 id="4">4. 参考文献 </h3>

* [Docker私有Registry 在 CentOS6.x 下安装指南](https://my.oschina.net/wstone/blog/355560#OSC_h3_14)

<h3 id="5">5. 附录</h3>

1. docker 1.8
2. registry 使用 docker registry V2，因为 docker registry V1 已经被官方废弃，并且不在维护。 同时在docker registry V2中支持OSS存储。
3. docker.xx.com 这是docker registry服务器的域名，也就是您公司docker私有服务器的主机 host，假设该host对应的ip地址为： 192.168.1.100；
4. nginx 1.9.4 被用来作为反向代理及负载均衡, 选择 nginx 1.9.4 的原因是 nginx 1.7.*后支持 add_header 等配置


