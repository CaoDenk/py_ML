from typing import List
import vtk
import torch
from rgbd_to_pointcloud import rgbd_to_pointcloud
from load_img import get_img_and_depth

from camera_args import load_camera_args

def vtk_show_points(l,rgb_colls,is_same_to_img:bool=False):
    
    points = vtk.vtkPoints()
    if is_same_to_img:       
        for i in l:
            points.InsertNextPoint(i[1],i[0],i[2])          
    else:
        for i in l:
            points.InsertNextPoint(i[0],i[1],i[2])
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)

    for r in rgb_colls:
        colors.InsertNextTuple3(r[0],r[1],r[2])
    # 创建vtkPolyData对象，并设置其点集和颜色为上面创建的vtkPoints对象和vtkUnsignedCharArray对象
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.GetPointData().SetScalars(colors)

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
    
    
