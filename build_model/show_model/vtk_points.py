import vtk

# 创建vtkPoints对象，并添加一些点
points = vtk.vtkPoints()
points.InsertNextPoint(0.0, 0.0, 0.0)
points.InsertNextPoint(1.0, 0.0, 0.0)
points.InsertNextPoint(0.0, 1.0, 0.0)
points.InsertNextPoint(0.0, 0.0, 1.0)

# 创建vtkPolyData对象，并设置其点集为上面创建的vtkPoints对象
poly_data = vtk.vtkPolyData()
poly_data.SetPoints(points)

# 创建vtkVertexGlyphFilter对象，将vtkPolyData对象中的点转换为顶点
vertex_filter = vtk.vtkVertexGlyphFilter()
vertex_filter.SetInputData(poly_data)
vertex_filter.Update()

# 创建vtkPolyDataMapper对象，并将vtkVertexGlyphFilter对象的输出连接到该对象的输入端口
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(vertex_filter.GetOutputPort())

# 创建vtkActor对象，并将vtkPolyDataMapper对象设置为该对象的Mapper
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# 创建vtkRenderer对象，并将vtkActor对象添加到该对象中
renderer = vtk.vtkRenderer()
renderer.AddActor(actor)

# 创建vtkRenderWindow对象，并将vtkRenderer对象添加到该对象中
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

# 创建vtkRenderWindowInteractor对象，并将vtkRenderWindow对象设置为该对象的交互窗口
interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# 开始交互式渲染
interactor.Initialize()
render_window.Render()
interactor.Start()