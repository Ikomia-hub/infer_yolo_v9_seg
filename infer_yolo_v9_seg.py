from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        # Instantiate algorithm object
        from infer_yolo_v9_seg.infer_yolo_v9_seg_process import InferYoloV9SegFactory
        return InferYoloV9SegFactory()

    def get_widget_factory(self):
        # Instantiate associated widget object
        from infer_yolo_v9_seg.infer_yolo_v9_seg_widget import InferYoloV9SegWidgetFactory
        return InferYoloV9SegWidgetFactory()
