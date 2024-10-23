
class C(): pass


def config(ev, debug=True, timeout=10, cross_check_threshold=0.15):
    c = C()
    c.debug = debug
    c.timeout = timeout
    c.cross_check_threshold = cross_check_threshold

    if ev == 'test':
        print(f'using environment test')

        # c.pcd3dBox = 'http://10.0.3.13:5000/pointcloud/3dbox'  # old test machine
        c.pcd3dBox = "http://10.3.0.171:5000/pointcloud/3dbox" 
        c.image2dBox = f'http://10.0.3.13:33339/yolov8/'
        c.fusion = 'http://10.3.0.171:27888/fusion/'
        c.track = f'http://10.3.0.171:27666/tracking_service/'
        
        return c
    elif ev == 'production':
        print(f'using environment production')
        
        return c