# 环境配置参考步骤
！！会覆盖原有容器
docker-compose up -d  
### 查看各容器ip
172.18.0.2 mdw
172.18.0.3 sdw1
172.18.0.4 sdw2
172.18.0.5 sdw3
### 打通各节点
（每个节点）更改root/gpadmin用户密码
（每个节点）免密操作 gpadmin用户
ssh-keygen -t rsa
ssh-copy-id -i ~/.ssh/id_rsa.pub root@mdw
ssh-copy-id -i ~/.ssh/id_rsa.pub root@sdw1
ssh-copy-id -i ~/.ssh/id_rsa.pub root@sdw2
ssh-copy-id -i ~/.ssh/id_rsa.pub root@sdw3
### 打通节点(mdw节点) 创建all_hosts和seg_hosts文件
gpssh-exkeys -f /home/gpadmin/all_hosts
→ 报错没有免密（实际可以操作）
    mdw节点重复ssh gpadmin@sdw1...后再次尝试
→报错cannot establish ssh access into the local host
    参考gpt检查sshd配置文件并修改，重新尝试后成功
### 测试连接
gpssh -f /home/gpadmin/all_hosts
gpssh -f /home/gpadmin/seg_hosts

master节点gpadmin账户，/home/gpadmin目录下
source /usr/local/greenplum-db-6.25.1/greenplum_path.sh
-s: segment(容器)的个数 -n: 每个segment(容器)上primary的个数
artifact/prepare.sh -s 3 -n 1
### 对应后面报错修改信息
vim gpinitsystem_config
### 新建文件夹
mkdir /home/gpadmin/data1
mkdir /home/gpadmin/data2
mkdir /home/gpadmin/data3
### 初始化集群，会生成env.sh 文件(greenplum所需的环境变量)
gpinitsystem -a -c gpinitsystem_config
→报错 Inconsistency between number of multi-home hostnames and number of segments per host
    (3segments)修改为 declare -a DATA_DIRECTORY=(  /home/gpadmin/data1  /home/gpadmin/data2 /home/gpadmin/data3)
→报错-Have lock file /tmp/.s.PGSQL.5432.lock but no process running on port 5432
    rm -f /tmp/.s.PGSQL.5432.lock
→卡在gpinitsystem:mdw:gpadmin-[INFO]:-Completed restart of Greenplum instance in production mode
    在其他节点登录mdw，运行gpstop -a
### 开启远程无密码访问
artifact/postinstall.sh
### 查看安装结果
ps -ef | grep postgres

### 问题：数据库无法连接  【~/.ssh/config不要乱改】
→似乎没有正确创建的数据库表
    编辑检查pg_hba.conf，确保包含以下配置：
    # TYPE  DATABASE        USER            ADDRESS                 METHOD
    local   all             gpadmin                                 trust
    host    all             gpadmin         127.0.0.1/32            trust
    host    all             gpadmin         ::1/128                 trust
    保存更改后，运行gpstop -u更新配置
检查日志tail -n 100 $MASTER_DATA_DIRECTORY/pg_log/gpseg-1.log，显示failed to acquire resources on one or more segments","could not connect to server: Connection refused
    运行gpstate -s检查段节点状态，netstat -plnt | grep 5432检查每个段节点端口（端口未开放）
段节点data1等目录为空
    赋予目录权限（sdw1）
    chown -R gpadmin:gpadmin /home/gpadmin/data1
    chown -R gpadmin:gpadmin /home/gpadmin/data2
    chown -R gpadmin:gpadmin /home/gpadmin/data3
    chmod 700 /home/gpadmin/data1
    chmod 700 /home/gpadmin/data2
    chmod 700 /home/gpadmin/data3
    尝试手动启动段节点服务：source /usr/local/greenplum-db-6.25.1/greenplum_path.sh
    pg_ctl -D /home/gpadmin/data1 -l /home/gpadmin/data1/pg_log/startup.log start
    pg_ctl -D /home/gpadmin/data2 -l /home/gpadmin/data2/pg_log/startup.log start
→报错pg_ctl: error while loading shared libraries: libssl.so.10: cannot open shared object file: No such file or directory
    echo "source /usr/local/greenplum-db-6.25.1/greenplum_path.sh" >> ~/.bashrc
    source ~/.bashrc
    sudo ln -s /usr/lib64/libssl.so.1.1.1f /usr/lib64/libssl.so.10
    sudo ln -s /usr/lib64/libcrypto.so.1.1.1f /usr/lib64/libcrypto.so.10
    尝试手动启动段节点服务
→报错pg_ctl: /lib64/libcrypto.so.10: version `libcrypto.so.10' not found (required by /usr/local/greenplum-db-6.25.1/lib/libpq.so.5)
    sudo yum install openssl-libs
    wget https://mirrors.aliyun.com/centos/8/AppStream/x86_64/os/Packages/compat-openssl10-1.0.2o-3.el8.x86_64.rpm
    sudo rpm -ivh compat-openssl10-1.0.2o-3.el8.x86_64.rpm --nodeps --force
    sudo ln -sf /usr/lib64/libssl.so.1.0.2o /usr/lib64/libssl.so.10
    sudo ln -sf /usr/lib64/libcrypto.so.1.0.2o /usr/lib64/libcrypto.so.10
    sudo ldconfig
    尝试手动启动段节点服务
→报错pg_ctl: directory "/home/gpadmin/data1" is not a database cluster directory
    初始化数据目录：initdb -D /home/gpadmin/data1 (data2 data3)
    → error while loading shared libraries: libnsl.so.1: cannot open shared object file: No such file or directory
    sudo yum install libnsl
    手动启动段节点服务成功（pg_ctl -D /home/gpadmin/data1 -l /home/gpadmin/data1/pg_log/startup.log start）

→ 在段节点运行netstat -plnt | grep 5432提示(No info could be read for "-p": geteuid()=1000 but you should be root.)
    加sudo
    段节点5432端口未开放
    pg_ctl: could not read file "/home/gpadmin/data1/postmaster.opts"
    
20240806:11:46:42:063869 gpstart:mdw:gpadmin-[INFO]:-DBID:3  FAILED  host:'mdw' datadir:'/home/gpadmin/data2/gpseg1' with reason:'Segment data directory does not exist for: '/home/gpadmin/data2/gpseg1''
20240806:11:46:42:063869 gpstart:mdw:gpadmin-[INFO]:-DBID:4  FAILED  host:'mdw' datadir:'/home/gpadmin/data3/gpseg2' with reason:'Segment data directory does not exist for: '/home/gpadmin/data3/gpseg2''
20240806:11:46:42:063869 gpstart:mdw:gpadmin-[INFO]:-DBID:2  FAILED  host:'mdw' datadir:'/home/gpadmin/data1/gpseg0' with reason:'Segment data directory does not exist for: '/home/gpadmin/data1/gpseg0''

 psql -d postgres -U gpadmin
 
-- 参考：https://www.jianshu.com/p/2f3d9925965d