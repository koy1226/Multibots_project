#ifndef TAB1DEVCONTROL_H
#define TAB1DEVCONTROL_H

#include <QWidget>
#include <QTimer>
#include <QDebug>
#include <QDial>
#include "keyled.h"

namespace Ui {
class Tab1DevControl;
}

class Tab1DevControl : public QWidget
{
    Q_OBJECT

public:
    explicit Tab1DevControl(QWidget *parent = nullptr);
    ~Tab1DevControl();
    QDial * getpQDial();

private:
    Ui::Tab1DevControl *ui;
    KeyLed * pKeyled;
    QTimer * pQTimer;
private slots:
    void progressBarSetSlot(int);
    void keyCheckBoxSlot(int);
    void setValueDialSlot();
    void timerStartSlot(bool);
    void timerValueChangedSlot(QString);
};

#endif // TAB1DEVCONTROL_H
