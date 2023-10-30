#ifndef TAB2CAM_H
#define TAB2CAM_H

#include <QWidget>
#include <QWebEngineView>
#include <QPixmap>
#include <QFileDialog>

QT_BEGIN_NAMESPACE
namespace Ui { class Tab2Cam; }
QT_END_NAMESPACE

class Tab2Cam : public QWidget
{
    Q_OBJECT

public:
    Tab2Cam(QWidget *parent = nullptr);
    ~Tab2Cam();

private slots:
    void captureCamView1();  // pPBCapture1 버튼 클릭 시 호출될 슬롯
    void captureCamView2();  // pPBCapture2 버튼 클릭 시 호출될 슬롯

private:
    Ui::Tab2Cam *ui;
    QWebEngineView  *pWebView00;
    QWebEngineView  *pWebView01;
};
#endif // TAB2CAM_H
