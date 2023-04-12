import vtk


if __name__ =='__main__':
    # 创建vtkSphereSource对象
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(1.0)  # 设置球面半径为1
    sphere.SetCenter(0.0, 0.0, 0.0)  # 设置球心坐标为(0, 0, 0)
    sphere.SetThetaResolution(32)  # 设置纬度方向上的分辨率
    sphere.SetPhiResolution(32)  # 设置经度方向上的分辨率

    # 创建vtkPolyDataMapper对象
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(sphere.GetOutputPort())

    # 创建vtkActor对象
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    # 创建vtkRenderer对象
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)

    # 创建vtkRenderWindow对象
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    # 创建vtkRenderWindowInteractor对象
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    # 开始交互式渲染
    interactor.Initialize()
    render_window.Render()
    interactor.Start()
