#include "tab2cam.h"
#include "ui_tab2cam.h"
#include <QDir> // 추가
#include <QMessageBox>

Tab2Cam::Tab2Cam(QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::Tab2Cam)
{
    ui->setupUi(this);
    pWebView00 = new QWebEngineView(this);
    pWebView00->load(QUrl(QStringLiteral("http://10.10.14.168:8080/?action=stream")));
    QVBoxLayout *layout1 = new QVBoxLayout;
    layout1->addWidget(pWebView00);
    ui->pCamView1->setLayout(layout1);

    pWebView01 = new QWebEngineView(this);
    pWebView01->load(QUrl(QStringLiteral("http://10.10.14.168:8081/?action=stream")));
    QVBoxLayout *layout2 = new QVBoxLayout;
    layout2->addWidget(pWebView01);
    ui->pCamView2->setLayout(layout2);

    // 버튼과 슬롯 연결
    connect(ui->pPBCapture1, &QPushButton::clicked, this, &Tab2Cam::captureCamView1);
    connect(ui->pPBCapture2, &QPushButton::clicked, this, &Tab2Cam::captureCamView2);

}

Tab2Cam::~Tab2Cam()
{
    delete ui;
}

void Tab2Cam::captureCamView1()
{
    // pWebView00의 화면 캡처
    QPixmap pixmap = pWebView00->grab();

    // "save_img" 디렉토리가 없으면 생성
    QDir().mkpath("/home/ubuntu/multibots/save_img");

    QString fileName = "/home/ubuntu/multibots/save_img/capture_front.jpg"; // 원하는 파일명으로 수정
    // 이미지 저장
    if (pixmap.save(fileName)) {
        // 캡처가 성공했을 때 메시지 출력
        QMessageBox::information(this, "캡처 완료", "캡처가 완료되었습니다.");
    } else {
        // 캡처 실패 시 메시지 출력
        QMessageBox::critical(this, "캡처 오류", "캡처를 저장하는 데 문제가 발생했습니다.");
    }
}

void Tab2Cam::captureCamView2()
{
    // pWebView01의 화면 캡처
    QPixmap pixmap = pWebView01->grab();

    // "save_img" 디렉토리가 없으면 생성
    QDir().mkpath("/home/ubuntu/multibots/save_img");

    QString fileName = "/home/ubuntu/multibots/save_img/capture_back.jpg"; // 원하는 파일명으로 수정
    // 이미지 저장
    if (pixmap.save(fileName)) {
        // 캡처가 성공했을 때 메시지 출력
        QMessageBox::information(this, "캡처 완료", "캡처가 완료되었습니다.");
    } else {
        // 캡처 실패 시 메시지 출력
        QMessageBox::critical(this, "캡처 오류", "캡처를 저장하는 데 문제가 발생했습니다.");
    }
}

