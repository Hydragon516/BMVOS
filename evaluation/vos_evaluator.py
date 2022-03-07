import os
import time
import torch
import utils
import torch.nn.functional as F


class VOSEvaluator(object):
    def __init__(self, dataset, save=False):
        self.dataset = dataset
        self.save = save
        self.imsavehlp = utils.ImageSaveHelper()
        if dataset.__class__.__name__ == 'DAVIS_Test':
            self.sdm = utils.ReadSaveDAVISChallengeLabels()
        elif dataset.__class__.__name__ == 'YTVOS_Test':
            self.sdm = utils.ReadSaveYTVOSChallengeLabels()

    def read_video_part(self, video_part):
        imgs = video_part['imgs'].cuda()
        given_masks = [mask.cuda() if mask is not None else None for mask in video_part['given_masks']]
        fnames = video_part['fnames']
        val_frame_ids = video_part['val_frame_ids']
        return imgs, given_masks, fnames, val_frame_ids

    def evaluate_video(self, model, seqname, video_parts, output_path, save):
        for video_part in video_parts:
            imgs, given_masks, fnames, val_frame_ids = self.read_video_part(video_part)
            original_imgs = imgs.clone()
            original_given_masks = given_masks.copy()

            tiny_check = False
            if imgs.size(-2) != 480 and imgs.size(-1) != 480:
                tiny_check = True
                H, W = imgs.size(-2), imgs.size(-1)
                if W > H:
                    ratio = 480 / H
                    imgs = F.interpolate(imgs[0], size=(480, int(ratio * W)), mode='bicubic', align_corners=False).unsqueeze(0)
                    for i in range(len(given_masks)):
                        if given_masks[i] is None:
                            continue
                        else:
                            given_masks[i] = F.interpolate(given_masks[i].float(), size=(480, int(ratio * W)), mode='nearest').long()
                else:
                    ratio = 480 / W
                    imgs = F.interpolate(imgs[0], size=(int(ratio * H), 480), mode='bicubic', align_corners=False).unsqueeze(0)
                    for i in range(len(given_masks)):
                        if given_masks[i] is None:
                            continue
                        else:
                            given_masks[i] = F.interpolate(given_masks[i].float(), size=(int(ratio * H), 480), mode='nearest').long()

            if tiny_check:
                tiny_obj = 0
                for i in range(len(given_masks)):
                    if given_masks[i] is None:
                        continue
                    else:
                        object_ids = given_masks[i].unique().tolist()
                        if 0 in object_ids:
                            object_ids.remove(0)
                        for obj_idx in object_ids:
                            obj_num = len(given_masks[i][given_masks[i] == obj_idx])
                            if obj_num < 1000:
                                tiny_obj += 1
                if tiny_obj > 0:
                    imgs = original_imgs
                    given_masks = original_given_masks

            t0 = time.time()
            tracker_out = model(imgs, given_masks, val_frame_ids)
            t1 = time.time()

            if save is True:
                for idx in range(len(fnames)):
                    fpath = os.path.join(output_path, seqname, fnames[idx])
                    data = ((tracker_out['segs'][0, idx, 0, :, :].cpu().byte().numpy(), fpath), self.sdm)
                    self.imsavehlp.enqueue(data)
        return t1-t0, imgs.size(1)

    def evaluate(self, model, output_path):
        model.cuda()
        model.eval()
        with torch.no_grad():
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            tot_time, tot_frames = 0.0, 0.0
            for seqname, video_parts in self.dataset.get_video_generator():
                savepath = os.path.join(output_path, seqname)
                if not os.path.exists(savepath):
                    os.makedirs(savepath)
                time_elapsed, frames = self.evaluate_video(model, seqname, video_parts, output_path, self.save)
                tot_time += time_elapsed
                tot_frames += frames
                print(seqname, 'fps:{}, frames:{}, time:{}'.format(frames / time_elapsed, frames, time_elapsed))
            print('\nTotal fps:{}\n\n'.format(tot_frames/tot_time))

        if self.save is True:
            self.imsavehlp.kill()
