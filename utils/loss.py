# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Loss functions
"""
import numpy as np
import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma    # g = 0
        if g > 0:  # 进不去
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions  p是模型预测输出结果，是list有3个元素，每个元素是yolo中一个检测头的输出tensor(16,3,80,80,7)、后面两个元素tensor中是40，40和20，20
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx  # b是target属于哪一个batch，for循环i=0的时候a就是第一层yolo中映射到特征图的3个anchor尺寸，gj、gi是target匹配到的某个anchor正样本的网格坐标
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj  （16，3，80，80）四维tensor值都是0

            n = b.shape[0]  # number of targets  # 一个batch16张图片的txt经过扩充数据集在第一个yolo检测头层上拥有的GT个数  debug的时候为 n=462
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions  从预测结果中分利用掩膜从16*3*80*80中提取出462个有对应GT的pred_bbox结果

                # Regression  box回归误差
                pxy = pxy.sigmoid() * 2 - 0.5  # anchor到pred_bbox需要的   移动因子
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]  # anchor到pred_bbox需要的  宽高缩放因子
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)   # -------- 计算预测pred_bbox和GT间的CIOU  iou是462的tensor向量
                lbox += (1.0 - iou).mean()  # iou loss  # ------------------------通过CIOU计算  模型预测结果和GT的 box回归误差loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:  # 进不去
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:  # 进不去
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio  # tobj在上面初始化为(16，3，80，80)是0  现在赋值为 pred_bbox和GT的iou

                # Classification   反正这一段就是求了 一下分类损失  用的二至交叉熵BCEloss
                if self.nc > 1:  # cls loss (only if multiple classes)    nc=2
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets  pcls是预测每个类别的得分(462,2) self.cn=0表示新建的t都填充为0，
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)  # 物体置信度损失也用BCEloss
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:  # 进不去  ---- 回到循环计算下一个 yolo检测头的box、obj、cls损失  循环3次
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:  # 进入去
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']  # 三个yolo层汇总得到的这个损失 × 超参数中的损失权重系数
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size = 16

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets  # na=3， nt=137
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain    gain=[1,1,1, 1,1,1,1]  7个1是tensor数组
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)  # ai是3行 137列二维tensor 第一行是137个0，第二行全为1，第三行全为2    repeat函数表示第0维度复制1次（没变），第1维度复制137次
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices  targets从(137,5)先变成(3,137,6) 再和 (3,137,1)的新ai变量在第2维度拼接 拼接后targets是(3,137,7)也就是比原来多了一列0   repeat这里先在第0维度复制3次（增加一个维度）
        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape  # anchors是(3,2)  shape是pred的结果形状(16,3,80,80,7)   anchors 是配置文件9个anchor分3组各自下采样8、16、32倍后的9个anchor 取self.anchors[0]就是下采样8的第一个yolo层的3个anchor，其特征图尺寸80*80
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain 将上行shape中第4，3，4，3个数据 赋值给gain的从第3到第6个位置的数 执行完后 gain=[1,1,80,80,80,80,1]只有7个元素的一维tensor

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)   # 将标注txt中的0-1xywh信息映射为特征图大小的GT_featuremap   ------ 去归一化
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio   r是（3，137，2）  3张137行2列的表：第一张表是targer第一层和第一个yolo层中3个中第一个anchor的比值表，第二、三层类推【(137,2)中第一列是GT_feature:前3个anchor映射到第一层yolo层的w之比   第二列是h之比】
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # j是(3,137)二维存放True、False的表  3行表示3个anchor；每一列是一个GT_feature  如果表的某行某列是True说明该GT和改行的anchor宽之比＜4，高之比也＜4
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter  t这个target变量从 (3,137,7) 经过本行变为 （230，7）二维数组了！！！！  ----------------t从(137,6)变为(230,7)是把有的GT匹配2个或3个anchor，该GT数据从1行变为2行或3行，这2两数据唯一的区别在于最后一列是0和1或2

                # Offsets
                gxy = t[:, 2:4]  # grid xy  # gxy是GT在特征图(如80*80)上中心点的坐标   如gxy为（46.7, 61.1）这个点，不过实际上gxy不止一个点，而是230个点的x和y
                gxi = gain[[2, 3]] - gxy  # inverse   # gxi 这个变量名字作者取得不好，实际上就是  特征图size减去上面的GT中心点坐标 得到的就是中心点到特征图右边、下边的距离 ！
                j, k = ((gxy % 1 < g) & (gxy > 1)).T  # j和k都是230个元素的True、False向量；j表示GT的中心点是否位于自身所在grid_cell的左半侧（x<0.5），k表示GT的中心点是否位于自身grid_cell的上半侧(y<0.5)
                l, m = ((gxi % 1 < g) & (gxi > 1)).T  # 同理，l判断GT的中心点是否位于右半侧（x>0.5）; m判断是否位于下半侧（y>0.5）   --- gxy>1和gxi>1 是去除80*80特征图的边缘一圈像素
                j = torch.stack((torch.ones_like(j), j, k, l, m))  # j将5个【230个元素的向量】拼成  (5，230) 二维tensor  第一行全为True，第二行是原来j的值，第三行是k，然后是l,m
                t = t.repeat((5, 1, 1))[j]  # t先从(230,7)变为(5,230,7)  经过[j]后变为(690,7)  实际上就是每个GT增加了周围2个新GT，所以t从230个增大3倍到690个了。增加的新GT可以是原GT左上、右上、左下、右下任意一组的两个新grid_cell
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors  # bc表示从第1到第690个GT各自所属的batch和class是什么(690,2)  gxy、gwh同理都是(690,2) a则是(690,1)
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class   # 将a变为（690）的向量 ，b和c从（b，c）中提取为一维 向量 都是(690)
            gij = (gxy - offsets).long()  # gij 就是GT中心点的向下取整   也就是所在grid_cell左上角坐标 【实际上是包括两次扩充正样本之后的点有690个】
            gi, gj = gij.T  # grid indices  # 从gij二维tensor中  单独提取 gi 和 gj 都是（690）的一维tensor

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid  # indices是一个list里面只有一个tuple元素，tuple中有四个tensor向量元素 4个tensor分别是所属batch、andhor、所在特征图的行数gj、列数gi
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box  tbox是list里面有一个tensor元素 这个tensor是(690,4)690表示扩充后有690个GT每行是 Δx，Δy，w，h四个数。需注意Δx和Δy的取值范围是（-0.5到1.5）! 690个GT顺序：前三分之一的GT（230个）都是本gridcell的GT，后面剩下的三分之二的GT随机分布在原grid_cell的左、上、右、下4个grid_cell中的1个，且顺序就是先把左边的都放一堆，然后是上面的以此类推
            anch.append(anchors[a])  # anchors  把690个GT匹配的anchor放入  anch  其中anch是个list，里面有一个tensor元素  这个tensor是（690，2）的二维张量， 2列分别是anchor的宽高【当然anchor是映射到特征图后大小的anchor】
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
        # tcls, tbox, indices, anch  四个返回值都是list 且list中都只有3个元素，分别记录每个检测头的相关信息
        # tcls是有3个元素的list，第1个元素是txt文件中GT匹配第一层的3个anchor扩充GT后的690个元素的tensor，第2个元素是匹配第二层yolo层的835个元素的tensor，第三个元素是匹配第三个检测头的anchor的420个元素的tensor
        # tbox同理也是[]list中有3个tensor元素，每个元素分别是对应3个yolo检测头，具体而言每个tensor中存放的是 Δx，Δy，w，h 其中Δx，Δy都是-0.5到1.5的取值范围，wh都是GT映射到特征图上后的大小