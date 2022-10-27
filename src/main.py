from database import Database
import json
import doc2vec


def main():
    f_config = open('./config.json', 'r')
    config_dic = json.load(f_config)
    f_config.close()

    db = Database(config_dic)
    db.scan_files()

    # if config_dic['ASM_2_VEC']:
    #     pass
    # else:
    #     doc2vec.train_model(config_dic)
    #     doc2vec.compare_sim(config_dic)


if __name__ == '__main__':
    main()
