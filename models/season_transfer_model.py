import torch
from util.util import tensor2im
from models.base_model import BaseModel
from torch.autograd import Variable

################## SeasonTransfer #############################
class SeasonTransferModel(BaseModel):
    def __init__(self, opt):
        BaseModel.__init__(self, opt)

    def prepare_data(self, data):
        img, attr_source, path_s = data
        img_A = img[0].to(self.device)
        img_B = img[1].to(self.device)
        source = attr_source[0].to(self.device)
        target = attr_source[1].to(self.device)
        self.current_data = [img_A, img_B, source, target]
        return self.current_data

    def translation(self, data):
        with torch.no_grad():
            self.prepare_data(data)
            img_A, img_B, _, _ = self.current_data
            style_B = self.enc_styleB(img_B)
            results_s2w, results_w2s = [('input_summer', tensor2im(img_A[0].data))], [
                ('input_winter', tensor2im(img_B[0].data))]
            fakes = self.genAB(img_A, style_B)
            results_s2w.append(('s2w_tone_{}'.format(1), tensor2im(fakes[0].data)))

            return results_s2w + results_w2s
