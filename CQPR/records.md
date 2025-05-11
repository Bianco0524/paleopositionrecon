# 部分实验结果输出控制台记录
部分PyGPlates和BPPR方法实验结果输出记录（表3.9 三种方法在不同重建时间的对比实验结果）
--------sample: 100000--------
Start reconstruct
total_time: 19.41
Start reconstruct
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
25/02/20 11:14:53 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
total_time: 14.31
--------sample: 200000--------
Start reconstruct
total_time: 35.46
Start reconstruct
total_time: 16.04
--------sample: 300000--------
Start reconstruct
total_time: 51.98
Start reconstruct
total_time: 22.02
--------sample: 400000--------
Start reconstruct
total_time: 68.35
Start reconstruct
25/02/20 11:18:30 WARN TaskSetManager: Stage 0 contains a task of very large size (1097 KiB). The maximum recommended task size is 1000 KiB.
total_time: 27.31
--------sample: 500000--------
Start reconstruct
total_time: 86.64
Start reconstruct
25/02/20 11:20:28 WARN TaskSetManager: Stage 0 contains a task of very large size (1369 KiB). The maximum recommended task size is 1000 KiB.
total_time: 34.18

部分BPPR方法实验结果输出记录（图3.6 三种方法在不同重建时间下重建百万级模拟数据集的实验结果）
-----sample: 1000000-----
Start reconstruct
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
25/02/21 12:26:37 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
25/02/21 12:26:45 WARN TaskSetManager: Stage 0 contains a task of very large size (5236 KiB). The maximum recommended task size is 1000 KiB.
total_time: 103.32
-----sample: 1500000-----
Start reconstruct
25/02/21 12:28:43 WARN TaskSetManager: Stage 0 contains a task of very large size (7947 KiB). The maximum recommended task size is 1000 KiB.
total_time: 149.67
-----sample: 2000000-----
Start reconstruct
25/02/21 12:31:29 WARN TaskSetManager: Stage 0 contains a task of very large size (10591 KiB). The maximum recommended task size is 1000 KiB.
total_time: 201.66
-----sample: 2500000-----
Start reconstruct
25/02/21 12:35:00 WARN TaskSetManager: Stage 0 contains a task of very large size (13302 KiB). The maximum recommended task size is 1000 KiB.
total_time: 255.88
-----sample: 3000000-----
Start reconstruct
25/02/21 12:39:30 WARN TaskSetManager: Stage 0 contains a task of very large size (16013 KiB). The maximum recommended task size is 1000 KiB
total_time: 339.55