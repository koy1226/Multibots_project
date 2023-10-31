#include "tab1db.h"
#include "ui_tab1db.h"
#include <QProcess>
#include <QFile>
#include <QTextStream>

Tab1DB::Tab1DB(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::Tab1DB)
{
    ui->setupUi(this);
    pQTimer = new QTimer(this);
    listWidget = new QListWidget(this);

    connect(ui->pPBappQuit, SIGNAL(clicked()), qApp, SLOT(quit()));

    // 체크박스 연결을 루프를 통해 처리
    for (int i = 1; i <= 37; ++i) {
        QCheckBox *checkBox = this->findChild<QCheckBox *>(QString("pCBkey%1").arg(i));
        if (checkBox) {
            connect(checkBox, &QCheckBox::stateChanged, [=](int state){
                QString objectName = checkBox->objectName();
                QString keyNumber = objectName.mid(6);
                int keyIndex = keyNumber.toInt() - 1;

                QString text = checkBox->text();

                if(state == Qt::Checked) {
                    qDebug() << QString("Key %1 is activated.").arg(keyIndex + 1);
                    ui->textEdit->append(text);
                    shoppinglist.append(text);
                } else if(state == Qt::Unchecked) {
                    qDebug() << QString("Key %1 is deactivated.").arg(keyIndex + 1);
                    shoppinglist.removeOne(text);

                    QTextCursor cursor(ui->textEdit->document());
                    while (!cursor.atEnd()) {
                        cursor = ui->textEdit->document()->find(text, cursor);

                        if (!cursor.isNull()) {
                            cursor.removeSelectedText();
                        } else {
                            break;
                        }
                    }
                }
            });
        }
    }

}
// 나머지 체크 박스들도 유사한 방식으로 처리

void Tab1DB::on_pPBClear_clicked()
{
    ui->textEdit->clear();
    ui->pCBkey1->setChecked(false);
    ui->pCBkey2->setChecked(false);
    ui->pCBkey3->setChecked(false);
    ui->pCBkey4->setChecked(false);
    ui->pCBkey5->setChecked(false);
    ui->pCBkey6->setChecked(false);
    ui->pCBkey7->setChecked(false);
    ui->pCBkey8->setChecked(false);
    ui->pCBkey9->setChecked(false);
    ui->pCBkey10->setChecked(false);
    ui->pCBkey11->setChecked(false);
    ui->pCBkey12->setChecked(false);
    ui->pCBkey13->setChecked(false);
    ui->pCBkey14->setChecked(false);
    ui->pCBkey15->setChecked(false);
    ui->pCBkey16->setChecked(false);
    ui->pCBkey17->setChecked(false);
    ui->pCBkey18->setChecked(false);
    ui->pCBkey19->setChecked(false);
    ui->pCBkey20->setChecked(false);
    ui->pCBkey21->setChecked(false);
    ui->pCBkey22->setChecked(false);
    ui->pCBkey23->setChecked(false);
    ui->pCBkey24->setChecked(false);
    ui->pCBkey25->setChecked(false);
    ui->pCBkey26->setChecked(false);
    ui->pCBkey27->setChecked(false);
    ui->pCBkey28->setChecked(false);
    ui->pCBkey29->setChecked(false);
    ui->pCBkey30->setChecked(false);
    ui->pCBkey31->setChecked(false);
    ui->pCBkey32->setChecked(false);
    ui->pCBkey33->setChecked(false);
    ui->pCBkey34->setChecked(false);
    ui->pCBkey35->setChecked(false);
    ui->pCBkey36->setChecked(false);
    ui->pCBkey37->setChecked(false);

}

void Tab1DB::on_pPBStart_clicked()
{
    // shoppinglist 내용을 텍스트 파일로 저장
    QFile file("/home/ubuntu/multibots/shoppinglist.txt");
    if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream out(&file);
        for (const QString &product : shoppinglist) {
            out << product << "\n";
        }
        file.close();
        qDebug() << "File saved successfully.";
    } else {
        qDebug() << "Failed to open shoppinglist.txt for writing.";
        qDebug() << "Error:" << file.errorString();
    }

    // 파이썬 스크립트 실행
    QProcess pythonProcess;
    pythonProcess.setWorkingDirectory("/home/ubuntu/multibots");
    pythonProcess.start("python3", QStringList() << "send2cart.py"); // 파이썬 스크립트 파일명을 제공하세요
    if (pythonProcess.waitForFinished()) {
        qDebug() << "Python script executed successfully.";QByteArray standardOutput = pythonProcess.readAllStandardOutput();
        QByteArray standardError = pythonProcess.readAllStandardError();
        qDebug() << "Standard Output:" << standardOutput;
        qDebug() << "Standard Error:" << standardError;
    } else {
        qDebug() << "Failed to execute Python script.";
    }
}

Tab1DB::~Tab1DB()
{
    delete ui;
}
