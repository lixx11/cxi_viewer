#!/usr/bin/env python3

import sys
import os

import pyqtgraph as pg
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import pyqtSlot
from PyQt5.uic import loadUi
from pyqtgraph.parametertree import Parameter

import h5py
import numpy as np
import yaml
from functools import partial
from settings import Settings
from stream_parser import Stream
from geometry import Geometry, det2fourier, get_hkl


class GUI(QtGui.QMainWindow):
    def __init__(self, settings):
        super(GUI, self).__init__()
        # load settings
        self.workdir = settings.workdir
        self.data_location = settings.data_location
        self.peak_info_location = settings.peak_info_location
        self.pixel_size = settings.pixel_size
        self.det_dist = settings.detector_distance
        self.cxi_file = settings.cxi_file
        self.geom_file = settings.geom_file
        self.ref_stream_file = settings.ref_stream_file
        self.test_stream_file = settings.test_stream_file

        # setup ui
        dir_ = os.path.dirname(os.path.abspath(__file__))
        loadUi('%s/layout.ui' % dir_, self)
        self.splitterH.setSizes([self.width() * 0.7, self.width() * 0.3])
        self.splitterV.setSizes([self.height() * 0.7, self.height() * 0.3])
        # add stream table window
        # self.streamTable = StreamTable(parent=self)

        self.nb_frame = 0
        self.frame = 0
        self.data = None
        self.geom = None
        self.ref_stream = None
        self.ref_events = []
        self.ref_reflections = {}
        self.test_stream = None
        self.test_events = []
        self.test_reflections = {}
        self.peaks = None
        self.nb_peaks = None
        self.show_hkl = False
        self.show_ref_stream = True
        self.show_test_stream = True
        self.rearrange = False

        # plot items
        self.peak_item = pg.ScatterPlotItem(
            symbol='x', size=10, pen='r', brush=(255, 255, 255, 0)
        )
        self.ref_reflection_item = pg.ScatterPlotItem(
            symbol='o', size=12, pen='g', brush=(255, 255, 255, 0)
        )
        self.test_reflection_item = pg.ScatterPlotItem(
            symbol='o', size=14, pen='y', brush=(255, 255, 255, 0)
        )
        self.peak_stream_item = pg.ScatterPlotItem(
            symbol='d', size=5, pen='r', brush=(255, 255, 255, 0)
        )
        self.ref_stream_item = pg.ScatterPlotItem(
            symbol='t', size=5, pen='g', brush=(255, 255, 255, 0)
        )
        self.test_stream_item = pg.ScatterPlotItem(
            symbol='t1', size=5, pen='y', brush=(255, 255, 255, 0)
        )

        self.image_view.getView().addItem(self.peak_item)
        self.image_view.getView().addItem(self.ref_reflection_item)
        self.image_view.getView().addItem(self.test_reflection_item)
        self.stream_plot.addItem(self.ref_stream_item)
        self.stream_plot.addItem(self.test_stream_item)
        self.stream_plot.addItem(self.peak_stream_item)

        # setup parameter tree
        params = [
            {
                'name': 'cxi file', 'type': 'str', 'value': self.cxi_file,
                'readonly': True
            },
            {
                'name': 'total frame', 'type': 'str',
                'value': None, 'readonly': True
            },
            {
                'name': 'geom file', 'type': 'str',
                'value': 'not set',
                'readonly': True
            },
            {
                'name': 'ref stream', 'type': 'str',
                'value': 'not set', 'readonly': True
            },
            {
                'name': 'test stream', 'type': 'str',
                'value': 'not set',
                'readonly': True
            },
            {
                'name': 'det dist', 'type': 'float', 'siPrefix': True,
                'suffix': 'mm', 'value': self.det_dist,
            },
            {
                'name': 'pixel size', 'type': 'float',
                'siPrefix': True, 'suffix': 'μm', 'value': self.pixel_size
            },
            {
                'name': 'current frame', 'type': 'int',
                'value': self.frame
            },
            {
                'name': 'show ref stream', 'type': 'bool',
                'value': self.show_ref_stream
            },
            {
                'name': 'show test stream', 'type': 'bool',
                'value': self.show_test_stream
            },
            {
                'name': 'rearrange', 'type': 'bool',
                'value': self.rearrange,
            }
        ]
        self.params = Parameter.create(name='params', type='group',
                                       children=params)
        self.parameterTree.setParameters(self.params, showTop=False)

        if self.cxi_file is not None:
            self.load_cxi(self.cxi_file)
        if self.geom_file is not None:
            self.load_geom(self.geom_file)
        if self.ref_stream_file is not None:
            self.load_stream(self.ref_stream_file, 'ref')
        if self.test_stream_file is not None:
            self.load_stream(self.test_stream_file, 'test')

        # menu bar actions
        self.action_load_cxi.triggered.connect(self.set_cxi)
        self.action_load_geom.triggered.connect(self.set_geom)
        self.action_load_ref_stream.triggered.connect(
            partial(self.load_stream, flag='ref'))
        self.action_load_test_stream.triggered.connect(
            partial(self.load_stream, flag='test')
        )

        # parameter tree slots
        self.params.param(
            'current frame').sigValueChanged.connect(self.change_frame)
        self.params.param(
            'show ref stream').sigValueChanged.connect(
            partial(self.change_show_stream, flag='ref')
        )
        self.params.param(
            'show test stream').sigValueChanged.connect(
            partial(self.change_show_stream, flag='test')
        )
        self.params.param(
            'rearrange').sigValueChanged.connect(
            self.change_rearrange
        )

        # image view / stream plot slots
        self.peak_stream_item.sigClicked.connect(self.stream_item_clicked)
        self.ref_stream_item.sigClicked.connect(self.stream_item_clicked)
        self.test_stream_item.sigClicked.connect(self.stream_item_clicked)

    @pyqtSlot()
    def set_cxi(self):
        filepath, _ = QtGui.QFileDialog.getOpenFileName(
            self, 'Open File', self.workdir, 'CXI File (*.cxi)')
        if filepath == '':
            return
        self.load_cxi(filepath)

    def load_cxi(self, cxi_file):
        try:
            h5_obj = h5py.File(cxi_file, 'r')
            data = h5_obj[self.data_location]
        except IOError:
            print('Failed to load %s' % cxi_file)
            return
        self.cxi_file = cxi_file
        self.data = data
        self.nb_frame = self.data.shape[0]
        # collect peaks from cxi, XPosRaw/YPosRaw is fs, ss, respectively
        peaks_x = h5_obj['%s/peakXPosRaw' % self.peak_info_location].value
        peaks_y = h5_obj['%s/peakYPosRaw' % self.peak_info_location].value
        nb_peaks = h5_obj['%s/nPeaks' % self.peak_info_location].value
        self.peaks = []
        self.nb_peaks = nb_peaks
        for i in range(len(nb_peaks)):
            self.peaks.append(
                np.concatenate(
                    [
                        peaks_y[i, :nb_peaks[i]].reshape(-1, 1),
                        peaks_x[i, :nb_peaks[i]].reshape(-1, 1)
                    ],
                    axis=1
                )
            )
        self.params.param('cxi file').setValue(self.cxi_file)
        self.params.param('total frame').setValue(self.nb_frame)
        self.update_display()
        self.update_stream_plot()

    def set_geom(self):
        filepath, _ = QtGui.QFileDialog.getOpenFileName(
            self, 'Open file', self.workdir, 'Geom File (*.geom *.h5 *.data)')
        if filepath == '':
            return
        self.load_geom(filepath)

    def load_geom(self, geom_file):
        geom_file = geom_file
        self.geom_file = geom_file
        self.params.param('geom file').setValue(self.geom_file)
        self.geom = Geometry(geom_file, pixel_size=self.pixel_size)

    @pyqtSlot(object, object)
    def change_frame(self, _, frame):
        if frame < 0:
            frame = 0
        elif frame > self.nb_frame:
            frame = self.nb_frame - 1
        self.frame = frame
        self.update_display()

    @pyqtSlot(str)
    def set_stream(self, flag):
        filepath, _ = QtGui.QFileDialog.getOpenFileName(
            self, 'Open File', self.workdir, 'Stream File (*.stream)')
        if filepath == '':
            return
        self.load_stream(filepath, flag)

    def load_stream(self, stream_file, flag):
        stream = Stream(stream_file)
        # collect reflections
        all_reflections = {}
        events = []
        for i in range(len(stream.chunks)):
            chunk = stream.chunks[i]
            event = chunk.event
            reflections = []
            if len(chunk.crystals) > 0:
                events.append(event)
            for j in range(len(chunk.crystals)):
                crystal = chunk.crystals[j]
                for k in range(len(crystal.reflections)):
                    reflection = crystal.reflections[k]
                    reflections.append(
                        [reflection.ss, reflection.fs]
                    )
            all_reflections[event] = np.array(reflections)
        if flag == 'ref':
            self.ref_stream_file = stream_file
            self.ref_stream = stream
            self.ref_reflections = all_reflections
            self.ref_events = events
            self.params.param('ref stream').setValue(self.ref_stream_file)
        elif flag == 'test':
            self.test_stream_file = stream_file
            self.test_stream = stream
            self.test_reflections = all_reflections
            self.test_events = events
            self.params.param('test stream').setValue(self.test_stream_file)
        else:
            print('Undefined flag: %s' % flag)
        self.update_stream_plot()

    @pyqtSlot(object, object)
    def stream_item_clicked(self, _, pos):
        event = int(pos[0].pos()[0])
        self.params.param('current frame').setValue(event)

    @pyqtSlot(object, object, str)
    def change_show_stream(self, _, show, flag):
        if flag == 'ref':
            self.show_ref_stream = not self.show_ref_stream
        elif flag == 'test':
            self.show_test_stream = not self.show_test_stream
        else:
            raise ValueError('Undefined flag: %s' % flag)
        self.update_display()

    @pyqtSlot(object, object)
    def change_rearrange(self, _, rearrange):
        if rearrange:
            if self.geom_file is None:
                self.params.param('rearrange').setValue(False)
            else:
                self.rearrange = True
                self.load_geom(self.geom_file)
        else:
            self.rearrange = False
        self.update_display()

    def update_stream_plot(self):
        if self.cxi_file is not None:
            pos = np.concatenate(
                [
                    np.arange(self.nb_frame).reshape(-1, 1),
                    np.zeros(self.nb_frame).reshape(-1, 1)
                ],
                axis=1
            )
            self.peak_stream_item.setData(pos=pos)
        if len(self.ref_events) > 0:
            pos = np.concatenate(
                [
                    np.array(self.ref_events).reshape(-1, 1),
                    np.ones(len(self.ref_events)).reshape(-1, 1)
                ],
                axis=1,
            )
            self.ref_stream_item.setData(pos=pos)
        if len(self.test_events) > 0:
            pos = np.concatenate(
                [
                    np.array(self.test_events).reshape(-1, 1),
                    np.ones(len(self.test_events)).reshape(-1, 1) * 2
                ],
                axis=1
            )
            self.test_stream_item.setData(pos=pos)

    def update_display(self):
        if self.data is None:
            return
        image = self.data[self.frame]
        if self.rearrange:
            image = self.geom.rearrange(image)
        self.image_view.setImage(
            image, autoRange=False, autoLevels=False,
            autoHistogramRange=False
        )

        # plot peaks
        self.peak_item.clear()
        if self.rearrange:
            peaks = self.geom.map(self.peaks[self.frame])
        else:
            peaks = self.peaks[self.frame]
        self.peak_item.setData(pos=peaks + 0.5)

        # plot reflections
        self.ref_reflection_item.clear()
        if self.show_ref_stream and self.frame in self.ref_events:
            reflections = self.ref_reflections[self.frame]
            if self.rearrange:
                reflections = self.geom.map(reflections)
            if len(reflections) > 0:
                self.ref_reflection_item.setData(pos=reflections + 0.5)
        self.test_reflection_item.clear()
        if self.show_test_stream and self.frame in self.test_events:
            reflections = self.test_reflections[self.frame]
            if len(reflections) > 0:
                self.test_reflection_item.setData(pos=reflections + 0.5)


def main():
    if len(sys.argv) > 1:
        print('using setting from %s' % sys.argv[1])
        with open(sys.argv[1], 'r') as f:
            settings_dict = yaml.load(f)
    else:
        settings_dict = {}
        print('using default settings')
    settings = Settings(settings_dict)
    app = QtGui.QApplication(sys.argv)
    win = GUI(settings)
    win.setWindowTitle("CXI Viewer")
    win.show()
    app.exec_()


if __name__ == '__main__':
    main()
