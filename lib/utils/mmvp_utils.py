import cv2
import trimesh
import numpy as np

import torch

mmvp_base_data_path = 'data/base_data/mmvp_data'


class InsoleModule():
    def __init__(self, basdir=None):
        self.basdir = basdir

        self.maskL = np.loadtxt(
            f'{mmvp_base_data_path}/essentials/insole_render/insoleMaskL.txt').astype(np.int32)
        self.maskR = np.loadtxt(
            f'{mmvp_base_data_path}/essentials/insole_render/insoleMaskR.txt').astype(np.int32)
        self.pixel_num = np.sum(self.maskL) + np.sum(self.maskR)
        self.maskImg = np.concatenate([self.maskL, self.maskR], axis=1) > 0.5

        # insole to smpl
        self.insole2smplR = np.load(
            f'{mmvp_base_data_path}/essentials/insole_render/insole2smplR.npy',
            allow_pickle=True).item()
        self.insole2smplL = np.load(
            f'{mmvp_base_data_path}/essentials/insole_render/insole2smplL.npy',
            allow_pickle=True).item()
        self.footIdsL = np.loadtxt(
            f'{mmvp_base_data_path}/essentials/insole_render/footL_ids.txt').astype(np.int32)
        self.footIdsR = np.loadtxt(
            f'{mmvp_base_data_path}/essentials/insole_render/footR_ids.txt').astype(np.int32)

        model_temp = trimesh.load(
            f'{mmvp_base_data_path}/essentials/smplify_essential/smpl_template.obj', process=False)
        self.v_template = np.array(model_temp.vertices)
        self.v_footL, self.v_footR = self.v_template[
            self.footIdsL, :], self.v_template[self.footIdsR, :]
        self.faces = np.array(model_temp.faces)

    def show_insole(self, data):
        press_dim, rows, cols = data.shape
        img = np.ones((rows, cols * 2), dtype=np.uint8)
        imgL = np.uint8(data[0] * 5)
        imgR = np.uint8(data[1] * 5)
        img[:, :imgL.shape[1]] = imgL
        img[:, img.shape[1] - imgR.shape[1]:] = imgR
        imgColor = cv2.applyColorMap(img, cv2.COLORMAP_HOT)
        imgColor[~self.maskImg, :] = [100, 100, 100]
        return imgColor

    def showNormalizedInsole(self, data):
        data = (data * 255).astype(np.uint8)
        press_dim, rows, cols = data.shape
        img = np.ones((rows, cols * 2), dtype=np.uint8)
        imgL = np.uint8(data[0])
        imgR = np.uint8(data[1])
        img[:, :imgL.shape[1]] = imgL
        img[:, img.shape[1] - imgR.shape[1]:] = imgR
        imgColor = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        imgColor[~self.maskImg, :] = [0, 0, 0]
        return imgColor

    def showContact(self, contact_label):
        rows, cols = contact_label.shape
        img_cont = np.zeros([rows, cols, 3])
        img_cont[self.maskImg, :] = [255, 255, 255]
        img_cont[contact_label > 0.5, :] = [0, 0, 255]
        return img_cont

    def visMaskedPressure(self, data):
        pressure_data = np.zeros_like(self.maskImg).astype(np.float32)
        pressure_data[self.maskImg] = data
        img = (pressure_data * 255).astype(np.uint8)
        imgColor = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        imgColor[~self.maskImg, :] = [0, 0, 0]
        return imgColor, pressure_data

    def visMaskedContact(self, data):
        cont_data = np.zeros_like(self.maskImg).astype(np.float32)
        cont_data[self.maskImg] = data
        imgColor = self.showContact(cont_data)
        return imgColor, cont_data

    def sigmoidNorm(self, insole, pixel_weight, avg=False):
        if not avg:
            pixel_weight = pixel_weight / self.pixel_num
        insole_norm = (insole - pixel_weight) / pixel_weight
        insole_norm = torch.sigmoid(
            torch.from_numpy(insole_norm)).detach().cpu().numpy()
        return insole_norm

    def sigmoidLogNorm(self, insole, pixel_weight, avg=False):
        if not avg:
            pixel_weight = pixel_weight / self.pixel_num
        insole_norm = insole / pixel_weight
        insole_norm = torch.sigmoid(torch.log10(
            torch.from_numpy(insole_norm))).detach().cpu().numpy()
        return insole_norm

    def maxNorm(self, insole, max_press):
        insole_norm = insole / max_press
        return insole_norm

    def press2Cont(self, insole, pixel_weight, th=0.7, avg=False):
        press_sigmoid = self.sigmoidNorm(insole, pixel_weight, avg=avg)
        cv2.imwrite('debug/sigmoid.png',
                    self.showNormalizedInsole(press_sigmoid))
        contact_label = np.zeros_like(press_sigmoid)
        contact_label[press_sigmoid > th] = 1
        contact_label = np.concatenate([contact_label[0], contact_label[1]],
                                       axis=1)
        return contact_label

    def getVertsPress(self, contact_label):
        # Left
        left_press = contact_label[0]
        left_smpl = np.zeros([self.footIdsL.shape[0]], dtype=np.float32)
        for i in range(self.footIdsL.shape[0]):
            ids = self.footIdsL[i]
            if str(ids) in self.insole2smplL.keys():
                tmp = self.insole2smplL[str(ids)]
                _data = left_press[tmp[0], tmp[1]]
                if _data.shape[0] != 0:
                    left_smpl[i] = np.sum(_data, axis=0)
        # Right
        right_press = contact_label[1]
        right_smpl = np.zeros([self.footIdsR.shape[0]], dtype=np.float32)
        for i in range(self.footIdsR.shape[0]):
            ids = self.footIdsR[i]
            if str(ids) in self.insole2smplR.keys():
                tmp = self.insole2smplR[str(ids)]
                _data = right_press[tmp[0], tmp[1]]
                if _data.shape[0] != 0:
                    right_smpl[i] = np.sum(_data, axis=0)

        smpl_cont = np.stack([left_smpl, right_smpl])
        smpl_cont[smpl_cont > 0.5] = 1
        return smpl_cont

    def visSMPLContImage(self, contact_label):
        imgL = self.visSMPLFootImage(self.v_footL,
                                     self.footIdsL,
                                     contact_label=contact_label[0])
        imgR = self.visSMPLFootImage(self.v_footR,
                                     self.footIdsR,
                                     contact_label=contact_label[1])
        img = np.concatenate([imgL, imgR], axis=1)
        return img

    def visSMPLFootImage(self,
                         v_foot,
                         footIds,
                         img_H=3300,
                         img_W=1100,
                         contact_label=None,
                         vert_color=None,
                         point_size=40):
        tex_color = [0, 139, 139]
        line_color = [0, 255, 0]
        x_col = img_W - (v_foot[:, 0] - np.min(v_foot[:, 0])) / (
            np.max(v_foot[:, 0]) - np.min(v_foot[:, 0])) * (img_W - 1) - 1
        x_row = img_H - (v_foot[:, 2] - np.min(v_foot[:, 2])) / (
            np.max(v_foot[:, 2]) - np.min(v_foot[:, 2])) * (img_H - 1) - 1

        img = np.ones(((img_H + 50), (img_W + 100), 3), dtype=np.uint8) * 255
        point = np.concatenate(
            [x_row.reshape([-1, 1]).astype(np.int32),
             x_col.reshape([-1, 1])],
            axis=1)

        for j in range(self.faces.shape[0]):
            x, y, z = self.faces[j]
            if x in footIds and y in footIds:
                xi = np.where(footIds == x)[0]
                yi = np.where(footIds == y)[0]
                img = cv2.line(img, (int(point[xi, 1]), int(point[xi, 0])),
                               (int(point[yi, 1]), int(point[yi, 0])),
                               (line_color[2], line_color[1], line_color[0]),
                               2)
            if z in footIds and y in footIds:
                zi = np.where(footIds == z)[0]
                yi = np.where(footIds == y)[0]
                img = cv2.line(img, (int(point[zi, 1]), int(point[zi, 0])),
                               (int(point[yi, 1]), int(point[yi, 0])),
                               (line_color[2], line_color[1], line_color[0]),
                               2)
            if z in footIds and x in footIds:
                zi = np.where(footIds == z)[0]
                xi = np.where(footIds == x)[0]
                img = cv2.line(img, (int(point[zi, 1]), int(point[zi, 0])),
                               (int(point[xi, 1]), int(point[xi, 0])),
                               (line_color[2], line_color[1], line_color[0]),
                               2)

        if contact_label is not None and vert_color is None:
            for i in range(point.shape[0]):
                x, y = point[i, 0], point[i, 1]
                _cont_label = contact_label[i]
                if _cont_label > 0.5:
                    v_color = [0, 0, 0]
                else:
                    v_color = [255, 255, 255]
                img = cv2.circle(
                    img, (int(y), int(x)), point_size,
                    (int(v_color[2]), int(v_color[1]), int(v_color[0])), -1)
                img = cv2.putText(img, f'{footIds[i]}', (int(y), int(x) + 25),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                  (tex_color[2], tex_color[1], tex_color[0]))
        elif contact_label is None and vert_color is not None:
            for i in range(point.shape[0]):
                x, y = point[i, 0], point[i, 1]
                v_color = vert_color[i, ::-1]
                img = cv2.circle(
                    img, (int(y), int(x)), point_size,
                    (int(v_color[2]), int(v_color[1]), int(v_color[0])), -1)
                img = cv2.putText(img, f'{footIds[i]}', (int(y), int(x) + 25),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                  (tex_color[2], tex_color[1], tex_color[0]))

        return img

    def visSMPLFootModel(self, contact_label):
        if contact_label.shape[0] != 6890:
            contact = np.zeros(6890)
            contact[self.footIdsL] = contact_label[0]
            contact[self.footIdsR] = contact_label[1]
        else:
            contact = contact_label
        hit_id = (contact == 1).nonzero()[0]

        _mesh = trimesh.Trimesh(vertices=self.v_template,
                                faces=self.faces,
                                process=False)
        _mesh.visual.vertex_colors = (191, 191, 191, 255)
        _mesh.visual.vertex_colors[hit_id, :] = (0, 255, 0, 255)

        return _mesh


def insole_gen(insole_data):
    insole_module = InsoleModule()

    insole = np.concatenate([insole_data[0], insole_data[1]], axis=1)
    insole = np.pad(insole, (2, 2), 'constant')
    insole = cv2.resize(insole, (insole.shape[1] * 10, insole.shape[0] * 10))
    insole = cv2.GaussianBlur(insole, (3, 3), 0)
    insole = cv2.GaussianBlur(insole, (3, 3), 0)
    insole = cv2.GaussianBlur(insole, (3, 3), 0)

    maskImg = np.concatenate([insole_module.maskL, insole_module.maskR],
                             axis=1).astype(np.float32)
    maskImg = np.pad(maskImg, (2, 2), 'constant')
    maskImg = cv2.resize(maskImg,
                         (maskImg.shape[1] * 10, maskImg.shape[0] * 10))
    maskImg = np.concatenate([
        maskImg[:, :int(maskImg.shape[1] / 2)],
        np.zeros([maskImg.shape[0], 10]), maskImg[:,
                                                  int(maskImg.shape[1] / 2):]
    ],
                             axis=1)

    insole = np.concatenate([
        insole[:, :int(insole.shape[1] / 2)],
        np.zeros([insole.shape[0], 10]), insole[:,
                                                int(insole.shape[1] / 2):]
    ],
                            axis=1)
    insole[:,
           int(insole.shape[1] / 2):(int(insole.shape[1] / 2) + 10)] = np.nan
    insole[insole < 1e-3] = np.nan
    insole[maskImg < 0.2] = np.nan
    insole[250:, 135:] = np.nan



    pixel_weight = np.sum(insole_data)
    contact_label = insole_module.press2Cont(insole_data, pixel_weight, th=0.7)
    contact_smpl_foot = insole_module.getVertsPress(insole_data)


    return contact_smpl_foot, insole_module.footIdsL, insole_module.footIdsR